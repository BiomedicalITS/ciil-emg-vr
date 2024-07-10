import socket
import select
import numpy as np
import time
import traceback
import logging
import os

import threading
from threading import Lock

from libemg.feature_extractor import FeatureExtractor

from nfc_emg.schemas import POSE_TO_NAME
from nfc_emg.utils import reverse_dict, map_cid_to_name
from nfc_emg import datasets, utils

from config import Config
from memory import Memory


def csv_reader(file_path: str, array: np.ndarray, lock: Lock):
    """
    This function is meant to be run in a separate thread.

    TODO: could also be a udp reader which generates its own timestamps, but maybe a bit worse
    for reproducibility afterwards.

    It reads a CSV file and appends the data to a numpy array which is shared with the main thread.

    Said numpy array is lock-protected.
    """
    array_len, array_width = array.shape

    # Wait for CSV file to be created
    file_dir = "/".join(file_path.split("/")[:-1])
    file_name = file_path.split("/")[-1]
    while file_name not in os.listdir(file_dir):
        time.sleep(0.1)

    print(f"csv_reader: Starting read loop for {file_path}")

    with open(file_path, "r") as c:
        while True:
            lines = c.readlines()
            if len(lines) == 0:
                # time.sleep(0.01)
                continue

            # print(f"csv_reader: Read {len(lines)} lines from {file_path}")

            new_array = np.fromstring(
                "".join(lines).replace("\n", ","), sep=","
            ).reshape(-1, array_width)

            if len(new_array) < array_len:
                new_array = np.vstack((array, new_array))

            lock.acquire()
            array[:] = new_array[-array_len:]
            lock.release()


def worker(
    config: Config,
    in_port: int,
    unity_in_port: int,
    out_port: int,
):
    """
    The MemoryManager worker is a UDP server responsible for receiving context from Unity and receiving state from the AdaptationManager.

    It parses said context, retrieves the corresponding data and predictions and generates the adaptation data.

    After generating new adaptation data, it is written to disk and a UDP message is sent via "out_port".

    The message is simply "WROTE"

    If a "Q" is received from Unity, the worker will shut down.
    """
    save_dir = config.paths.get_experiment_dir()
    memory_dir = config.paths.get_memory()
    data_dir = config.paths.get_train()

    logger = logging.getLogger("memory_manager")
    fh = logging.FileHandler(save_dir + "mem_manager.log", mode="w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    # receive messages from the adaptation manager
    in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    in_sock.bind(("localhost", in_port))

    # send messages to this port (input of adaptManager)
    out_port = ("localhost", out_port)

    # receive context from unity
    unity_in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    unity_in_sock.bind(("localhost", unity_in_port))

    # Create some initial memory data
    base_odh = datasets.get_offline_datahandler(
        data_dir,
        utils.get_cid_from_gid(config.paths.gestures, data_dir, config.gesture_ids),
        utils.get_reps(data_dir),
    )
    base_win, base_labels = datasets.prepare_data(base_odh, config.sensor)
    base_features = FeatureExtractor().extract_features(
        config.features, base_win, array=True
    )
    memory = Memory().add_memories(
        base_features,
        np.eye(len(config.gesture_ids))[base_labels],
        np.zeros((len(base_labels), 3)),
        ["P"] * len(base_labels),
        [0.0] * len(base_labels),
    )

    # runtime constants
    name_to_cid = reverse_dict(map_cid_to_name(config.paths.get_train()))
    unity_to_cid_map = {k: name_to_cid[v] for k, v in POSE_TO_NAME.items() if v != -1}

    """
    Shared between worker and csv reader thread. CSV reader reads data from file and writes it to this array.
    CSV reader also takes care of rolling the data, meaning the newest data is always at the end of the array.
    """
    live_data = np.zeros(
        (
            config.sensor.fs // config.sensor.window_increment,
            np.prod(config.sensor.emg_shape) * config.sensor.window_size + 2,
        )
    )
    live_data_lock = Lock()
    threading.Thread(
        target=csv_reader,
        args=(
            config.paths.get_live() + "preds.csv",
            live_data,
            live_data_lock,
        ),
        daemon=True,
    ).start()

    start_time = time.perf_counter()

    print("MemoryManager is starting!")

    num_written = 0
    total_samples_unfound = 0
    is_adapt_mngr_waiting = False
    done = False
    while not done:
        try:
            ready_to_read, _, _ = select.select([in_sock, unity_in_sock], [], [], 0)

            if len(ready_to_read) == 0:
                time.sleep(0.01)
                continue

            for sock in ready_to_read:
                sock: socket.socket
                udp_packet = sock.recv(1024).decode()
                if sock == unity_in_sock:
                    if udp_packet == "Q":
                        # Unity sends "Q" when it shuts down / is done
                        done = True
                        in_sock.sendto(b"STOP", out_port)
                        del_t = time.perf_counter() - start_time
                        logger.info(f"MEMORYMANAGER: GOT DONE FLAG AT {del_t:.2f} s")
                    elif not (udp_packet.startswith("P") or udp_packet.startswith("N")):
                        # ensure context packet
                        continue

                    # New context received from Unity
                    logger.info(f"MEMORYMANAGER: GOT PACKET: {udp_packet}")

                    live_data_lock.acquire()
                    adap_data = live_data.copy()
                    live_data_lock.release()

                    if 0 in adap_data[:, 0]:
                        logger.info("MemoryManager waiting for more CSV data...")
                        continue

                    # logger.info(
                    #     f"Average time between timestamps: {np.mean(np.diff(adap_timestamps))*1000:.3f} ms"
                    # )

                    result = decode_unity(
                        udp_packet,
                        adap_data,
                        config.features,
                        config.sensor.window_size,
                        len(config.gesture_ids),
                        unity_to_cid_map,
                        config.negative_method,
                    )

                    if result is None:
                        total_samples_unfound += 1
                        logger.warning("No valid window found.")
                        continue

                    (
                        adap_data,
                        adap_label,
                        adap_possibilities,
                        adap_type,
                        timestamp,
                    ) = result

                    # print(udp_packet)
                    # print(adap_label)
                    # print(adap_possibilities)
                    # print(adap_type)
                    # print("=" * 50)

                    if len(adap_data) != len(adap_label):
                        continue

                    memory.add_memories(
                        adap_data,
                        adap_label,
                        adap_possibilities,
                        adap_type,
                        timestamp,
                    )
                    logger.info(f"MemoryManager: memory length {len(memory)}")

                if sock == in_sock or is_adapt_mngr_waiting:
                    if udp_packet == "WAITING":
                        # Training pass done so update model
                        is_adapt_mngr_waiting = True

                    if not is_adapt_mngr_waiting:
                        continue

                    if len(memory) == 0:
                        # don't write empty memory
                        continue

                    t1 = time.perf_counter()
                    memory.write(memory_dir, num_written)
                    del_t = time.perf_counter() - t1

                    logger.info(
                        f"MEMORYMANAGER: WROTE FILE: {num_written},\t lines:{len(memory)},\t unfound: {total_samples_unfound},\t WRITE TIME: {del_t:.2f}s"
                    )
                    num_written += 1
                    memory = Memory()

                    in_sock.sendto(b"WROTE", out_port)
                    is_adapt_mngr_waiting = False
        except Exception:
            print("MEMORYMANAGER: " + traceback.format_exc())
            in_sock.sendto(b"STOP", out_port)
            logger.error("MEMORYMANAGER: " + traceback.format_exc())
            return


def decode_unity(
    packet: str,
    data: np.ndarray,
    features: list,
    window_size: int,
    num_classes: int,
    unity_to_cid_map: dict,
    negative_method: str,
):
    """
    Decode a context packet from Unity. Only 1 valid window should be found.

    Returns None if no valid window is found.

    Returns:
        - features: np.ndarray with shape (1, L) where L is the # of features
        - label: np.ndarray with shape (1, n_classes), one-hot encoded label for adaptation
        - possibilities: np.ndarray with shape (1, 3), eg [0, 3, -1]. Padded to 3 elements with -1
        - outcome: ["P"] if model prediction was within-context, else ["N"]
        - timestamp: [time.time()] of the window
    """
    # Extract context...
    message_parts = packet.split(" ")
    outcome = message_parts[0]
    timestamp = float(message_parts[1])
    possibilities = [
        unity_to_cid_map[p] for p in message_parts[2:] if p in unity_to_cid_map
    ]
    # print(message_parts)

    # Now extract corresponding prediction and data window...
    pred_index = np.argwhere(data[:, 0] == timestamp)
    if len(pred_index) == 0:
        return None
    pred = int(data[pred_index, 1])
    windows = data[pred_index, 2:].reshape(1, -1, window_size)
    feats = FeatureExtractor().extract_features(features, windows, array=True)

    adaptation_label = np.zeros((1, num_classes))
    if outcome == "P":
        if pred not in possibilities:
            return None
        # within-context, use the prediction as-is
        adaptation_label[:, pred] = 1
    elif outcome == "N":
        if negative_method == "mixed":
            adaptation_label[:, possibilities] = 1 / len(possibilities)
        else:
            return None
        # dunno what happens if adaptation label left at zeros

    if len(possibilities) < 3:
        # pad to len 3 with -1
        possibilities += [-1] * (3 - len(possibilities))

    return (
        feats,
        adaptation_label,
        np.array([possibilities]),
        [outcome],
        [timestamp],
    )
