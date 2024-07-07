import copy
import socket
import select
import numpy as np
import time
import traceback
import logging
import threading
from threading import Lock

from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows

from nfc_emg.models import EmgCNN
from nfc_emg.schemas import POSE_TO_NAME
from nfc_emg.utils import reverse_dict, map_cid_to_name

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
    with open(file_path, "r") as c:
        while True:
            lines = c.readlines()
            if len(lines) == 0:
                time.sleep(0.01)
                continue

            new_array = np.fromstring(
                "".join(lines).replace("\n", ","), sep=","
            ).reshape(-1, array_width)

            if len(new_array) < array_len:
                new_array = np.vstack((array, new_array))

            lock.acquire()
            array[:] = new_array[-array_len:]
            lock.release()


def worker(config: Config, in_port: int, unity_in_port: int, out_port: int):
    """
    The MemoryManager worker is a UDP server responsible for receiving context from Unity and receiving state from the AdaptationManager.

    It parses said context, retrieves the corresponding data and predictions and generates the adaptation data.

    After generating new adaptation data, it is written to disk and a UDP message is sent via "out_port".

    The message is simply "WROTE"

    If a "Q" is received from Unity, the worker will shut down.
    """
    save_dir = config.paths.model.replace("model.pth", "")

    logging.basicConfig(
        filename=save_dir + "memorymanager.log",
        filemode="w",
        encoding="utf-8",
        level=logging.INFO,
    )

    # this is where we receive commands from the adaptation manager
    adaptation_manager_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    adaptation_manager_sock.bind(("localhost", in_port))

    # this is where we receive context from unity
    unity_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    unity_sock.bind(("localhost", unity_in_port))

    memory = Memory()
    fe = FeatureExtractor()

    name_to_cid = reverse_dict(map_cid_to_name(config.paths.train))
    unity_to_cid_map = {k: name_to_cid[v] for k, v in POSE_TO_NAME.items() if v != -1}

    """
    Shared between worker and csv reader thread. CSV reader reads data from file and writes it to this array.
    CSV reader also takes care of rolling the data, meaning the newest data is always at the end of the array.
    """
    live_data = np.zeros((3 * config.sensor.fs, np.prod(config.input_shape) + 1))
    live_data_lock = Lock()

    threading.Thread(
        target=csv_reader,
        args=(
            config.paths.live_data + "EMG.csv",
            config.input_shape,
            live_data,
            live_data_lock,
        ),
        daemon=True,
    ).start()

    start_time = time.perf_counter()

    num_written = 0
    total_samples_unfound = 0
    waiting_flag = 0
    done = False
    while not done:
        try:
            ready_to_read, _, _ = select.select(
                [adaptation_manager_sock, unity_sock], [], [], 0
            )
            for sock in ready_to_read:
                sock: socket.socket
                received_data, _ = sock.recvfrom(1024)
                udp_packet = received_data.decode("utf-8")

                if sock is unity_sock:
                    if udp_packet == "Q":
                        # Unity sends "Q" when it shuts down / is done
                        done = True
                        del_t = time.perf_counter() - start_time
                        logging.info(f"MEMORYMANAGER: GOT DONE FLAG AT {del_t:.2f}s")
                        continue

                    # New context received from Unity, fetch latest data
                    live_data_lock.acquire()
                    adap_data = live_data.copy()
                    live_data_lock.release()

                    # Get a copy of the model every time, since it's being adapted
                    # TODO does this work?
                    model = copy.deepcopy(config.model).to("cpu").eval()

                    # Decode data into timestamps and features arrays
                    # TODO maybe rework if not performant enough
                    # TODO adap_timestamps is not the same size as adap_windows
                    adap_timestamps = adap_data[:, 0]
                    adap_windows = get_windows(
                        adap_data[:, 1:],
                        config.sensor.window_size,
                        config.sensor.window_increment,
                    )
                    adap_features = fe.extract_features(
                        config.features, adap_windows, array=True
                    )
                    result = decode_unity(
                        udp_packet,
                        adap_timestamps,
                        adap_features,
                        model,
                        unity_to_cid_map,
                        "mixed",
                    )

                    if result is None:
                        total_samples_unfound += 1
                        continue

                    (
                        adap_data,
                        adap_labels,
                        adap_possibilities,
                        adap_type,
                        timestamp,
                    ) = result

                    if (len(adap_data)) != (len(adap_labels)):
                        continue

                    memory.add_memories(
                        adap_data,
                        adap_labels,
                        adap_possibilities,
                        adap_type,
                        timestamp,
                    )

                elif sock is adaptation_manager_sock or waiting_flag:
                    # waiting flag is set when AdaptationManager is done with an adaptation pass
                    if udp_packet != "WAITING" and not waiting_flag:
                        continue
                    waiting_flag = 1
                    if len(memory) == 0:
                        # don't write empty memory
                        continue
                    t1 = time.perf_counter()
                    memory.write(save_dir, num_written)
                    del_t = time.perf_counter() - t1
                    logging.info(
                        f"MEMORYMANAGER: WROTE FILE: {num_written},\t lines:{len(memory)},\t unfound: {total_samples_unfound},\t WRITE TIME: {del_t:.2f}s"
                    )
                    num_written += 1
                    memory = Memory()
                    adaptation_manager_sock.sendto(
                        "WROTE".encode("utf-8"), ("localhost", out_port)
                    )
                    waiting_flag = 0
        except Exception:
            logging.error("MEMORYMANAGER: " + traceback.format_exc())


def decode_unity(
    packet: str,
    timestamps: np.ndarray,
    feature_data: np.ndarray,
    model: EmgCNN,
    unity_to_cid_map: dict,
    negative_method: str,
):
    """

    Decode a context packet from Unity. Only 1 valid window should be found.

    Params:
        - timestamps with shape (n,)
        - feature_data: data shape (n, c) where c is the emg shape
    """
    message_parts = packet.split(" ")

    outcome = message_parts[0]
    timestamp = float(message_parts[1])
    possibilities = [unity_to_cid_map[p] for p in message_parts[2:]]

    # find the features: data - timestamps, <- features ->
    # TODO Load data here.... maybe pass in a file handle?
    feature_data_index = np.argwhere(timestamps == timestamp)
    if len(feature_data_index) == 0:
        return None

    # get the model prediction
    features = feature_data[feature_data_index]
    prediction = model.predict(features)

    # Create the adaptation label
    adaptation_label = np.zeros(model.classifier.out_features)
    if outcome == "P":
        # within-context, use the prediction as-is
        try:
            adaptation_label[int(prediction)] = 1
        except Exception:
            return None
    elif outcome == "N":
        if negative_method == "mixed":
            mixed_label = np.zeros(model.classifier.out_features)
            mixed_label[possibilities] = 1 / len(possibilities)

    # TODO save the model's prediction too?
    timestamp = [timestamp]
    adaptation_outcome = np.array(outcome)
    adaptation_direction = np.array(possibilities)

    return (
        features,
        adaptation_label,
        adaptation_direction,
        adaptation_outcome,
        timestamp,
    )
