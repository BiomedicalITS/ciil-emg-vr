import csv
import socket
import select
import numpy as np
import time
import traceback
import logging
import os
from collections import deque

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

    window_queue = deque(maxlen=array_len)
    lines_read = 0

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
                time.sleep(0.01)
                continue
            lines_read += len(lines)
            # print(f"csv_reader: Read {len(lines)} lines from {file_path}")

            new_array = np.fromstring("".join(lines).replace("\n", ","), sep=",")
            print(
                f"csv_reader: Read {len(lines)} new lines (total {lines_read}) with shape: {new_array.shape}"
            )
            new_array = new_array.reshape(-1, array_width)
            window_queue.extend(new_array)

            if len(window_queue) == array_len:
                with lock:
                    array[:] = np.array(window_queue)


def run_memory_manager(
    config: Config,
    unity_in_port: int,
    server_port: int,
):
    """
    The MemoryManager worker is a UDP server responsible for receiving context from Unity and receiving state from the AdaptationManager.

    It parses said Unity context, finds the corresponding data window and prediction and re-computes the features.

    After generating new adaptation data, it is written to disk and a UDP message is sent via "out_port" to tell the AdaptationManager.

    The message is simply "WROTE"

    If a "Q" is received from Unity, the worker will shut down.
    """
    save_dir = config.paths.get_experiment_dir()
    memory_dir = config.paths.get_memory()
    data_dir = config.paths.get_train()

    logger = logging.getLogger("memory_manager")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(save_dir + "mem_manager.log", mode="w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    fs = logging.StreamHandler()
    fs.setLevel(logging.INFO)
    logger.addHandler(fs)

    manager_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    manager_sock.bind(("localhost", server_port))

    adapt_manager_addr = None

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

    # memory = Memory()

    # runtime constants
    name_to_cid = reverse_dict(map_cid_to_name(config.paths.get_train()))
    unity_to_cid_map = {k: name_to_cid[v] for k, v in POSE_TO_NAME.items() if v != -1}

    """
    Shared between worker and csv reader thread. CSV reader reads data from file and writes it to this array.
    CSV reader also takes care of rolling the data: the newest data is at the end of the array.
    """
    live_data = np.zeros(
        (
            config.sensor.fs // config.sensor.window_increment,
            np.prod(config.sensor.emg_shape) * config.sensor.window_size + 2,
        )
    )
    live_data_lock = Lock()
    csv_t = threading.Thread(
        target=csv_reader,
        args=(
            config.paths.get_live() + "preds.csv",
            live_data,
            live_data_lock,
        ),
        daemon=True,
    )
    csv_t.start()

    start_time = time.perf_counter()

    logger.info("MM: starting")

    num_written = 0
    total_samples_unfound = 0
    is_adapt_mngr_waiting = False
    done = False
    while not done:
        try:
            if not csv_t.is_alive():
                logger.error("CSV THREAD DEADDDDDDDDDDd")
                raise Exception("CSV reader thread died")

            ready_to_read, _, _ = select.select([manager_sock, unity_in_sock], [], [])

            if len(ready_to_read) == 0:
                time.sleep(0.05)
                continue

            for sock in ready_to_read:
                sock: socket.socket
                udp_packet, address = sock.recvfrom(1024)
                udp_packet = udp_packet.decode()
                if sock == unity_in_sock:
                    # Unity sends "Q" when it shuts down / is done
                    if udp_packet == "Q":
                        done = True
                        manager_sock.sendto(b"STOP", adapt_manager_addr)
                        del_t = time.perf_counter() - start_time
                        logger.info(f"MM: done flag at {del_t:.2f} s")
                        continue
                    # ensure context packet
                    elif not (udp_packet.startswith("P") or udp_packet.startswith("N")):
                        continue

                    logger.info(f"MM: {udp_packet}")

                    with live_data_lock:
                        adap_data = live_data.copy()

                    # Timestamp is only 0 if the array is not full [timestamp, prediction, *data]
                    if 0 in adap_data[:, 0]:
                        logger.info("MM: waiting for more CSV data")
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
                        logger.warning("MM: no matching window found")
                        continue

                    (
                        adap_data,
                        adap_label,
                        adap_possibilities,
                        adap_was_pred_good,
                        timestamp,
                    ) = result

                    if len(adap_data) != len(adap_label):
                        logger.error(
                            "MM: Adaptation data and adaptation label length mismatch"
                        )
                        continue

                    memory.add_memories(
                        adap_data,
                        adap_label,
                        adap_possibilities,
                        adap_was_pred_good,
                        timestamp,
                    )
                    logger.info(f"MM: memory len {len(memory)}")

                if sock == manager_sock or is_adapt_mngr_waiting:
                    if adapt_manager_addr is None:
                        adapt_manager_addr = address
                        logger.info(
                            f"MM: Adaptation manager address set to {adapt_manager_addr}"
                        )

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
                        f"MM: write #{num_written} with len {len(memory)}, unfound: {total_samples_unfound}, WRITE TIME: {del_t:.2f} s"
                    )
                    num_written += 1
                    memory = Memory()

                    manager_sock.sendto(b"WROTE", adapt_manager_addr)
                    is_adapt_mngr_waiting = False
        except Exception as e:
            logger.error(f"MM: {e}")
        finally:
            manager_sock.sendto(b"STOP", adapt_manager_addr)
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
    outcome = message_parts[0]  # "P" for positive, "N" for negative
    timestamp = float(message_parts[1])  # timestamp sent from classifier
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
