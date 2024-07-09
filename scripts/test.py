from multiprocessing import Process
from re import T
import time
import os

import numpy as np

from libemg.feature_extractor import FeatureExtractor
from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier

from nfc_emg.sensors import EmgSensorType, EmgSensor
from nfc_emg.utils import get_online_data_handler
from nfc_emg import datasets, utils, models, schemas as s
from nfc_emg.paths import NfcPaths

import configs as g


def test_oclassi():
    SENSOR = EmgSensorType.BioArmband

    sensor = EmgSensor(SENSOR, window_inc_ms=25)
    sensor.start_streamer()

    paths = NfcPaths(f"data/{sensor.get_name()}")
    paths.set_trial_number(paths.trial - 1)
    paths.gestures = "data/gestures/"
    paths.set_model_name("model")

    name_to_cid = utils.reverse_dict(utils.map_cid_to_name(paths.train))
    unity_to_cid_map = {k: name_to_cid[v] for k, v in s.POSE_TO_NAME.items() if v != -1}

    print(unity_to_cid_map)

    # model = models.load_mlp(paths.model)
    model = models.EmgSCNNWrapper.load_from_disk(paths.model, sensor.emg_shape, "cuda")
    model.__setattr__("classes_", list(range(len(g.FUNCTIONAL_SET))))

    odh = utils.get_online_data_handler(
        sensor,
        False,
        timestamps=True,
        file=True,
        file_path=paths.live_data,
    )
    model.model.eval()
    # odh.analyze_hardware()
    time.sleep(10)
    classi = EMGClassifier()
    classi.add_majority_vote(sensor.maj_vote_n)
    classi.classifier = model

    oclassi = OnlineEMGClassifier(
        classi, sensor.window_size, sensor.window_increment, odh, ["MAV"], std_out=False
    )
    oclassi.run(block=False)
    oclassi.analyze_classifier()


def test_rand():
    sensor = EmgSensor(EmgSensorType.BioArmband)
    paths = NfcPaths(f"data/{sensor.get_name()}")
    paths.set_trial_number(paths.trial - 1)

    classes = utils.map_gid_to_cid(paths.gestures, paths.train)

    # sensor.start_streamer()

    # odh = get_online_data_handler(
    #     sensor.fs,
    #     imu=False,
    #     attach_filters=(
    #         False if sensor.sensor_type == EmgSensorType.BioArmband else True
    #     ),
    # )
    # odh.visualize_channels(list(range(8)), 6000)

    odh = datasets.get_offline_datahandler(
        paths.train, list(classes.values()), utils.get_reps(paths.train)
    )
    windows, labels = datasets.prepare_data(odh, sensor)

    fe = FeatureExtractor()
    features = fe.extract_feature_group("HTD", windows)
    print("Allo")


def test_write():
    SENSOR = EmgSensorType.BioArmband

    sensor = EmgSensor(SENSOR, window_inc_ms=25)
    sensor.start_streamer()

    paths = NfcPaths(f"data/vr_{sensor.get_name()}")
    paths.set_trial_number(paths.trial - 1)
    paths.gestures = "data/gestures/"

    time.sleep(10)

    odh = utils.get_online_data_handler(
        sensor,
        False,
        timestamps=True,
        file=True,
        file_path=paths.live_data,
    )
    odh.options["file"] = False

    time.sleep(10)

    print("Setting file write to True")
    odh.options["file"] = True

    time.sleep(10)


def test_read_emg_csv():
    SENSOR = EmgSensorType.BioArmband
    sensor = EmgSensor(SENSOR, window_inc_ms=25)

    paths = NfcPaths(f"data/0/{sensor.get_name()}")
    # paths.set_trial_number(paths.trial_number - 1)
    paths.set_trial_number(0)
    paths.gestures = "data/gestures/"

    t0 = time.time()
    data = np.loadtxt(paths.live_data + "EMG.csv", delimiter=",").reshape(-1, 9)
    print(f"{time.time()-t0:.3f} s to load and match")

    dt = np.diff(data[:, 0])

    print(f"Loaded {len(dt)}, which is {len(dt) / sensor.fs:.3f} s of data")
    print(
        f"mean {np.mean(dt)*1000:.3f} std {np.std(dt)*1000:.3f}, fs {1/np.mean(dt):.3f} Hz"
    )


def test_online_rw():
    from collections import deque

    SENSOR = EmgSensorType.BioArmband
    sensor = EmgSensor(SENSOR, window_inc_ms=25)
    sensor.start_streamer()

    paths = NfcPaths(f"data/vr_{sensor.get_name()}")
    paths.set_trial_number(paths.trial - 1)
    paths.gestures = "data/gestures/"

    for csv in os.listdir(f"{paths.base}/{paths.trial}"):
        if not csv.startswith("live_"):
            continue
        os.remove(f"{paths.base}/{paths.trial}/{csv}")

    odh = utils.get_online_data_handler(
        sensor.fs,
        sensor.bandpass_freqs,
        sensor.notch_freq,
        False,
        False if SENSOR == EmgSensorType.BioArmband else True,
        timestamps=True,
        file=True,
        file_path=paths.live_data,
    )

    while "live_EMG.csv" not in os.listdir(f"{paths.base}/{paths.trial}"):
        continue

    t0 = time.time()

    ft_data = np.zeros((0, 8))
    line_q = deque(maxlen=sensor.fs)

    last_timestamp = 0
    timestamp = time.time() * 2
    with open(paths.live_data + "EMG.csv", "r") as csv:
        while True:
            time.sleep(0.1)

            timestamp = time.time()

            newlines = csv.readlines()
            newlines = np.fromstring(
                "".join(newlines).replace("\n", ","), sep=","
            ).reshape(-1, 9)
            print(
                f"Time to decode {len(newlines)} newlines: {time.time()-timestamp:.5f} s"
            )

            line_q.extend(newlines)
            print(f"Time to extend queue: {time.time()-timestamp:.5f} s")

            data = np.array(line_q)
            if data.size == 0:
                continue

            stamps = data[:, 0]
            valid_samples = np.squeeze(
                np.argwhere(
                    np.logical_and(stamps > last_timestamp, stamps <= timestamp)
                )
            )

            data = data[valid_samples, 1:]
            ft_data = np.vstack((ft_data, data))

            last_timestamp = timestamp

            print(f"{data.shape} {ft_data.shape} ({ft_data.nbytes/1024:.0f} kB)")
            print(f"fs {len(ft_data)/(time.time()-t0):.2f}")
            print(f"Total iter time: {time.time()-timestamp:.3f} s")


def test_dict_iter():
    fe = FeatureExtractor().get_feature_groups()["LS4"]

    dico = dict()

    for f in fe:
        dico[f] = np.zeros((10, 8))

    dico["a"] = "e"

    for feat in dico:
        print(feat)

    # Seems like dicts (CPython) iterate in order of key insertion.


def test_np_shared():
    """
    Conclusion from this test:

    Pre-determine the np array size. Do NOT use out-of-place operations on the array.
    """
    from multiprocessing import shared_memory
    from threading import Thread, Lock

    data = np.zeros((5, 9))
    new_data = np.ones((10, 9))

    if len(new_data) < len(data):
        new_data = np.vstack((data, new_data))

    data[:] = new_data[-len(data) :]
    print(data)

    exit()

    def write_process(arr: np.ndarray, lock: Lock):
        i = 0
        while True:
            new_data = np.zeros(1) + i

            lock.acquire()
            tmp_arr = np.hstack((arr, new_data))
            tmp_arr = np.roll(tmp_arr, -1, axis=0)[: len(arr)]
            arr[:] = tmp_arr
            lock.release()
            i += 1
            time.sleep(0.1)

    # shm_buf = shared_memory.SharedMemory(create=True, size=1024)
    # arr = np.ndarray(buffer=shm_buf.buf, dtype=np.int32, shape=(shm_buf.size,))
    lock = Lock()
    arr = np.zeros(10)
    print(arr.shape)

    Thread(target=write_process, args=(arr, lock), daemon=True).start()

    for i in range(len(arr)):
        lock.acquire()
        arr[i] = i * 1000
        print(f"Main: {arr}")
        lock.release()
        time.sleep(0.5)


def test_np():
    a = [0, 1, 2, 3, 4, 3]
    a = np.array(a)
    print(a.shape)
    print(np.argwhere(a == 3))

    from sklearn.preprocessing import OneHotEncoder

    ohe = OneHotEncoder()
    i = ohe.fit_transform(a.reshape(-1, 1))
    print(i)

    i = np.eye(5)[a]
    print(i)

    possibilities = [2, 3, 7, 0]
    mixed_label = np.zeros(8)
    mixed_label[possibilities] = 1 / len(possibilities)
    print(mixed_label)


def worker(lis):
    lis[0] = 0


def test_process():
    my_list = [1, 2, 3]

    Process(target=worker, args=(my_list,)).start()
    time.sleep(1)
    print(my_list)


def test_pid():
    from subprocess import check_output

    def get_pid(name):
        return check_output(["pidof", name])

    get_pid("python")


if __name__ == "__main__":
    # test_pid()
    # test_process()
    # test_np_shared()
    test_np()
    # test_dict_iter()
    # test_dict_iter()
    # test_read_emg_csv()
    # test_online_rw()
    # test_write()
    # test_oclassi()

    print("Exiting test")
