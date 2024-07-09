from copy import deepcopy
import socket
import select
import numpy as np
import time
import traceback
import logging
import os
import sys

# from multiprocessing import Process, shared_memory
# from multiprocessing import Lock

import threading
from threading import Lock

from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows

from config import Config
from nfc_emg.models import EmgCNN
from nfc_emg.schemas import POSE_TO_NAME
from nfc_emg.utils import reverse_dict, map_cid_to_name

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
    parent_dir = "/".join(file_path.split("/")[:-1])
    while "live_EMG.csv" not in os.listdir(parent_dir):
        time.sleep(0.1)
    print("csv_reader: Starting read loop")
    with open(file_path, "r") as c:
        while True:
            lines = c.readlines()
            if len(lines) == 0:
                time.sleep(0.01)
                continue

            # print(len(lines))

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
    model_lock: Lock,
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
    memory_dir = save_dir + "memory/"

    logger = logging.getLogger("memory_manager")
    fh = logging.FileHandler(save_dir + "mem_manager.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    # receive messages from the adaptation manager
    in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    in_sock.bind(("localhost", in_port))

    # receive context from unity
    unity_in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    unity_in_sock.bind(("localhost", unity_in_port))

    memory = Memory()

    name_to_cid = reverse_dict(map_cid_to_name(save_dir + "/train/"))
    unity_to_cid_map = {k: name_to_cid[v] for k, v in POSE_TO_NAME.items() if v != -1}

    """
    Shared between worker and csv reader thread. CSV reader reads data from file and writes it to this array.
    CSV reader also takes care of rolling the data, meaning the newest data is always at the end of the array.
    """
    live_data = np.zeros((3 * config.sensor.fs, np.prod(config.sensor.emg_shape) + 1))
    live_data_lock = Lock()

    threading.Thread(
        target=csv_reader,
        args=(
            save_dir + "/live_EMG.csv",
            live_data,
            live_data_lock,
        ),
        daemon=True,
    ).start()

    start_time = time.perf_counter()

    model_lock.acquire()
    model = deepcopy(config.model).to("cuda").eval()
    model_lock.release()

    num_written = 0
    total_samples_unfound = 0
    is_adapt_mngr_waiting = False
    done = False
    while not done:
        try:
            # TODO make sure we don't run the loop if the Unity context timestamp has not changed
            # TODO probably want to batch a bunch of context
            ready_to_read, _, _ = select.select([in_sock, unity_in_sock], [], [], 0)

            if len(ready_to_read) == 0:
                time.sleep(0.01)
                continue

            for sock in ready_to_read:
                sock: socket.socket
                udp_packet = sock.recv(1024).decode("utf-8")
                if sock == unity_in_sock:
                    if udp_packet == "Q":
                        # Unity sends "Q" when it shuts down / is done
                        done = True
                        del_t = time.perf_counter() - start_time
                        logger.info(f"MEMORYMANAGER: GOT DONE FLAG AT {del_t:.2f} s")
                        return
                    elif not (udp_packet.startswith("P") or udp_packet.startswith("N")):
                        # ensure context packet
                        continue

                    logger.info(f"MEMORYMANAGER: GOT PACKET: {udp_packet}")

                    # New context received from Unity, fetch latest data
                    live_data_lock.acquire()
                    adap_data = live_data.copy()
                    live_data_lock.release()

                    # Decode data into timestamps and features arrays
                    # TODO must manually filter fetched data
                    adap_timestamps = adap_data[:, 0]
                    if 0.0 in adap_timestamps:
                        logger.info("MemoryManager waiting for more CSV data...")
                        continue

                    # logger.info(
                    #     f"Average time between timestamps: {np.mean(np.diff(adap_timestamps))*1000:.3f} ms"
                    # )

                    result = decode_unity(
                        udp_packet,
                        adap_timestamps,
                        adap_data[:, 1:],
                        config.features,
                        config.sensor.window_size,
                        config.sensor.window_increment,
                        model,
                        unity_to_cid_map,
                        config.negative_method,
                    )

                    if result is None:
                        total_samples_unfound += 1
                        logger.warning("No valid window found.")
                        continue

                    (
                        adap_data,
                        adap_labels,
                        adap_possibilities,
                        adap_type,
                        timestamp,
                    ) = result

                    if len(adap_data) != len(adap_labels):
                        continue

                    memory.add_memories(
                        adap_data,
                        adap_labels,
                        adap_possibilities,
                        adap_type,
                        timestamp,
                    )
                    logger.info(f"MemoryManager: memory length {len(memory)}")

                if sock == in_sock or is_adapt_mngr_waiting:
                    if udp_packet == "WAITING":
                        # Training pass done so update model
                        model_lock.acquire()
                        model = deepcopy(config.model).to("cuda").eval()
                        model_lock.release()
                        is_adapt_mngr_waiting = True
                    elif udp_packet == "ERROR":
                        return

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
                    in_sock.sendto("WROTE".encode("utf-8"), ("localhost", out_port))
                    is_adapt_mngr_waiting = False
        except KeyboardInterrupt:
            in_sock.sendto("STOP".encode("utf-8"), ("localhost", out_port))
            return
        except Exception:
            logger.error("MEMORYMANAGER: " + traceback.format_exc())


def decode_unity(
    packet: str,
    timestamps: np.ndarray,
    data: np.ndarray,
    features: list,
    window_size: int,
    window_increment: int,
    model: EmgCNN,
    unity_to_cid_map: dict,
    negative_method: str,
):
    """
    Decode a context packet from Unity. Only 1 valid window should be found.

    Params:
        - timestamps with shape (n,)
        - feature_data: data shape (n, c) where c is the emg shape

    Returns None if no valid window is found.

    Returns:
        - features: np.ndarray with shape (1, L) where L is the # of features
        - label: np.ndarray with shape (1, n_classes), one-hot encoded label for adaptation
        - possibilities: np.ndarray with shape (1, 4), eg [0, 3, 4, -1]. Padded to 4 elements with -1
        - outcome: ["P"] if model prediction was within-context, else ["N"]
        - timestamp: [time.time()] of the window
    """
    message_parts = packet.split(" ")
    print(message_parts)

    outcome = message_parts[0]
    timestamp = float(message_parts[1])
    possibilities = [unity_to_cid_map[p] for p in message_parts[2:]]

    diff = np.abs(timestamps - timestamp)
    # receive classifier timestamp + timestamped data.
    # find the closest timestamp to the classifier timestamp and use it as "newest" data for window
    feature_data_index = np.argmin(diff)
    print(
        f"Unity decode: idx {feature_data_index}, dt {diff[feature_data_index]*1000:.3f} ms"
    )

    start = feature_data_index + 1 - window_size

    if start < 0:
        # TODO what if start is < 0?
        return None

    data = data[start : start + window_size]
    windows = get_windows(data, window_size, window_increment)
    features = FeatureExtractor().extract_features(features, windows, array=True)

    prediction = model.predict(features)
    adaptation_label = np.zeros((len(prediction), model.classifier.out_features))
    if outcome == "P":
        # within-context, use the prediction as-is
        adaptation_label[:, int(prediction.item(0))] = 1
    elif outcome == "N":
        if negative_method == "mixed":
            adaptation_label[:, possibilities] = 1 / len(possibilities)

    if len(possibilities) < 4:
        # pad with -1 to len 4
        possibilities += [-1] * (4 - len(possibilities))

    return (
        features,
        adaptation_label,
        np.array([possibilities]),
        [outcome],
        [timestamp],
    )
