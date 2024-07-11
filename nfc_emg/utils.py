import os
from typing import Iterable
import json
import shutil
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

import numpy as np

from libemg.data_handler import OnlineDataHandler
from libemg.screen_guided_training import ScreenGuidedTraining
from libemg.filtering import Filter

from nfc_emg.paths import NfcPaths
from nfc_emg.sensors import EmgSensor, EmgSensorType


def get_most_recent_checkpoint(lightning_logs_path: str = "./lightning_logs") -> str:
    max = 0
    for folder in os.listdir(lightning_logs_path):
        if "version" not in folder:
            continue

        with open(f"{lightning_logs_path}/{folder}/hparams.yaml", "r") as f:
            text = f.readline()
            if text.startswith("{}"):
                continue

        num = int(folder.split("_")[1])
        if num > max:
            max = num
    ckpt_name = os.listdir(f"{lightning_logs_path}/version_{max}/checkpoints")[0]
    model_ckpt_path = f"{lightning_logs_path}/version_{max}/checkpoints/{ckpt_name}"

    return model_ckpt_path


def get_reps(path: str):
    """
    Get all recorded repetitions from a directory (R_*)
    """
    return list(set([int(f.split("_")[1]) for f in os.listdir(path) if "R_" in f]))


def reverse_dict(d: dict):
    """
    Reverse a dictionary (key: value -> value: key)
    """
    return {v: k for k, v in d.items()}


def map_gid_to_name(gesture_img_dir: str, gids=None):
    """Map all gesture IDs (GID) to their human-readable name

    Args:
        - gesture_img_dir (str): path to LibEMG gestures
        - gids: only map these GIDs. If None, map all GIDs

    Returns a dictionary mapping the gesture ID to the human-readable gesture name

    """
    with open(gesture_img_dir + "gesture_list.json", "r") as f:
        gid_to_name: dict = json.load(f)
    ret = {}
    gid_to_name["17"] = "Wrist_Down"
    gid_to_name["18"] = "Wrist_Up"
    for k, v in gid_to_name.items():
        if not isinstance(v, str):
            continue
        if gids is None or int(k) in gids:
            ret[int(k)] = v
    return ret


def map_cid_to_name(data_dir: str, cids=None):
    """
    Map the ODH Class ID (CID) to the human-readable gesture name.

    Params:
        - data_dir: path to the directory where data is stored
        - cids: only map these CIDs. If None, map all CIDs
    Returns a dictionary mapping the class index to the human-readable gesture name
    """
    with open(data_dir + "metadata.json", "r") as f:
        metadata: dict = json.load(f)
    class_to_name = {}
    for val in metadata.values():
        if not isinstance(val, dict):
            continue
        if "class_idx" not in val and "class_name" not in val:
            continue
        if cids is None or val["class_idx"] in cids:
            if val["class_name"] == "OK":
                val["class_name"] = "Wrist_Down"
            if val["class_name"] == "Stop":
                val["class_name"] = "Wrist_Up"
            class_to_name[val["class_idx"]] = val["class_name"]
    return class_to_name


def map_gid_to_cid(gesture_img_dir: str, data_dir: str, gids=None):
    """
    Map LibEMG Gesture ID (GIDs) to ODH Class ID (CIDs)

    Args:
        - gesture_img_dir (str): where the images are stored
        - data_dir (str): where the data is stored
        - gids: only map these GIDs. If None, map all GIDs

    Returns a dictionary mapping the gesture ID to the class ID
    """
    gid_to_name = map_gid_to_name(gesture_img_dir, gids)
    name_to_gid = {}
    for g, n in gid_to_name.items():
        if not isinstance(n, str):
            continue
        name_to_gid[n] = g

    cid_to_name = map_cid_to_name(data_dir)
    gid_to_cid = {}
    for c, n in cid_to_name.items():
        if n not in gid_to_name.values():
            continue
        gid_to_cid[name_to_gid[n]] = c

    return gid_to_cid


def map_cid_to_ordered_name(gesture_img_dir: str, data_dir: str, gids=None):
    """
    Map ODH Class ID (CID) to the gesture name from GIDs.

    Args:
        gesture_img_dir (str): _description_
        data_dir (str): _description_
        gids (_type_, optional): _description_. Defaults to None.

    Returns: a dict of int[str]

    >>> map_cid_to_ordered_name(paths.gestures, paths.train, GESTURE_IDS)
    {
        0: 'Chuck_Grip',
        1: 'Hand_Close',
        2: 'Hand_Open',
        3: 'Index_Extension',
        4: 'Index_Pinch',
        5: 'No_Motion',
        6: 'Wrist_Extension',
        7: 'Wrist_Flexion'
    }
    """
    return map_cid_to_name(
        data_dir,
        get_cid_from_gid(gesture_img_dir, data_dir, gids),
    )


def get_cid_from_gid(gesture_img_dir: str, data_dir: str, gestures: list):
    """Get the ODH Class ID (CID) from a list of GIDs

    Args:
        - gesture_img_dir (str): where the images are stored
        - data_dir (str): where the data is stored
        - gids: only map these GIDs. If None, map all GIDs
    Returns a list of class indices
    """
    return list(map_gid_to_cid(gesture_img_dir, data_dir, gestures).values())


def get_name_from_gid(gestures_img_dir: str, data_dir: str, gestures: list):
    """From a list of GIDs, get the ordered list (CID) of gesture names.

    For example, this can be used to get the human-readable gesture name directly from a model prediction trained with data from `data_dir`.

    >>> class_names = get_name_from_gid(gestures_img_dir, data_dir, gestures)
    >>> pred = model.predict(data) # pred is a list of class indices
    >>> print([class_names[p] for p in pred])
    Hand Open
    Hand Close
    ...
    """
    return [
        map_cid_to_name(data_dir)[i]
        for i in sorted(get_cid_from_gid(gestures_img_dir, data_dir, gestures))
    ]


def get_filter(
    sampling_rate: float,
    bandpass_freqs: float | Iterable = (20, 450),
    notch_freq: int = 50,
):
    if not isinstance(bandpass_freqs, Iterable):
        bandpass_freqs = [bandpass_freqs]
    bandpass_freqs = list(bandpass_freqs)

    fi = Filter(sampling_frequency=sampling_rate)
    fi.install_filters({"name": "notch", "cutoff": notch_freq, "bandwidth": 3})
    if len(bandpass_freqs) == 2 and sampling_rate > bandpass_freqs[1] * 2:
        fi.install_filters(
            filter_dictionary={"name": "bandpass", "cutoff": bandpass_freqs, "order": 4}
        )
    else:
        fi.install_filters(
            {"name": "highpass", "cutoff": bandpass_freqs[0], "order": 2}
        )
    return fi


def get_online_data_handler(
    sensor: EmgSensor,
    imu: bool = True,
    **kwargs,
) -> OnlineDataHandler:
    """Get odh and install filters on it. Start listening to the stream.

    Args:
        kwargs: passed to OnlineDataHandler creator

    Returns:
        OnlineDataHandler, with listening activated
    """
    odh = OnlineDataHandler(imu_arr=imu, **kwargs)
    if sensor.sensor_type != EmgSensorType.BioArmband:
        odh.install_filter(
            get_filter(sensor.fs, sensor.bandpass_freqs, sensor.notch_freq)
        )
    odh.start_listening()
    return odh


def do_sgt(
    sensor: EmgSensor,
    gestures_list: list,
    gestures_dir: str,
    data_dir: str,
    num_reps: int,
    rep_time: int,
):
    sensor.start_streamer()
    odh = get_online_data_handler(sensor, False)
    screen_guided_training(
        odh, gestures_list, gestures_dir, num_reps, rep_time, data_dir
    )
    odh.stop_listening()
    sensor.stop_streamer()


def screen_guided_training(
    odh: OnlineDataHandler,
    gestures_id_list: list,
    gestures_img_dir: str,
    num_reps: int,
    rep_time: int,
    out_data_dir: str,
    **kwargs,
):
    """
    Do Screen Guided Training and save the data to disk.

    Params:
        - odh: ODH to use for data sampling
        - gestures_id_list: See [LibEMG Gestures](https://github.com/libemg/LibEMGGestures)
        - gestures_img_dir: directory where the images are stored
        - num_reps: number of repetitions per gesture
        - rep_time: time in seconds for each repetition
        - out_data_dir: directory where the data will be stored
        - kwargs: additional parameters for the training UI, passed as key-values to `ScreenGuidedTraining.launch_training`
    """
    to_download = []  # by id
    to_remove = []  # by name
    try:
        downloaded_gestures = [
            f.split(".")[0]
            for f in os.listdir(gestures_img_dir)
            if not f.endswith(".json")
        ]
        for id, name in map_gid_to_name(gestures_img_dir).items():
            if name in downloaded_gestures and id not in gestures_id_list:
                to_remove.append(name)
            elif name not in downloaded_gestures and id in gestures_id_list:
                to_download.append(id)
        # remove gestures not requested
        for name in to_remove:
            os.remove(gestures_img_dir + name)
    except Exception:
        # remove all gestures and redownload everything
        shutil.rmtree(gestures_img_dir, ignore_errors=True)
        to_download = gestures_id_list
    train_ui = ScreenGuidedTraining()
    train_ui.download_gestures(to_download, gestures_img_dir)
    if 17 in gestures_id_list:
        os.remove(gestures_img_dir + "OK.png")
        shutil.copy("wrist_images/OK.png", gestures_img_dir)
    if 18 in gestures_id_list:
        os.remove(gestures_img_dir + "Stop.png")
        shutil.copy("wrist_images/Stop.png", gestures_img_dir)
    if 24 in gestures_id_list:
        os.remove(gestures_img_dir + "Ring_Flexion.png")
        shutil.copy("wrist_images/Ring_Flexion.png", gestures_img_dir)

    train_ui.launch_training(
        odh,
        num_reps,
        rep_time,
        gestures_img_dir,
        output_folder=out_data_dir,
        # wait_btwn_prompts=True,
        **kwargs,
    )


def save_eval_results(results: dict, path: str):
    """
    Save evaluation results from LibEMG OfflineMetrics
    """
    with open(path, "w") as f:
        tmp_results = results.copy()
        tmp_results["CONF_MAT"] = 0
        json.dump(tmp_results, f, indent=4)
    return tmp_results


def show_conf_mat(results: dict, paths: NfcPaths, gesture_ids: list):
    """
    Show confusion matrix from results returned from LibEMG OfflineMetrics
    """
    conf_mat = results["CONF_MAT"] / np.sum(results["CONF_MAT"], axis=1, keepdims=True)
    test_gesture_names = get_name_from_gid(
        paths.gestures, paths.get_train(), gesture_ids
    )
    ConfusionMatrixDisplay(conf_mat, display_labels=test_gesture_names).plot()
    plt.show()


if __name__ == "__main__":
    screen_guided_training(
        OnlineDataHandler(), [1, 2, 3, 4, 5], "data/gestures/", 5, 5, "data/train/"
    )
