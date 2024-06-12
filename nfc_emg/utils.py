import os
from typing import Iterable
import json
import shutil

from libemg.data_handler import OnlineDataHandler
from libemg.screen_guided_training import ScreenGuidedTraining
from libemg.filtering import Filter


__BASE_DIR = "data/"
__TRAIN_DATA_DIR = __BASE_DIR + "train/"
__FINETUNE_DATA_DIR = __BASE_DIR + "finetune/"
__MODEL_PATH = __BASE_DIR + "model.pth"
__GESTURES_DIR = __BASE_DIR + "gestures/"


def set_paths(ext: str):
    """Set experiment paths

    Returns train data dir, finetune data dir, model path, gestures dir
    """
    global __BASE_DIR, __TRAIN_DATA_DIR, __FINETUNE_DATA_DIR, __MODEL_PATH, __GESTURES_DIR
    __BASE_DIR = f"data/{ext}/"
    __TRAIN_DATA_DIR = __BASE_DIR + "train/"
    __FINETUNE_DATA_DIR = __BASE_DIR + "finetune/"
    __MODEL_PATH = __BASE_DIR + "model.pth"
    __GESTURES_DIR = __BASE_DIR + "gestures/"
    return get_paths()


def get_paths():
    """Get paths

    Returns train data dir, finetune data dir, model path, gestures dir
    """
    global __BASE_DIR, __TRAIN_DATA_DIR, __FINETUNE_DATA_DIR, __MODEL_PATH, __GESTURES_DIR
    return __TRAIN_DATA_DIR, __FINETUNE_DATA_DIR, __MODEL_PATH, __GESTURES_DIR


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


def map_gid_to_name(gesture_img_dir: str):
    """Map all gesture IDs (GID) to their human-readable name

    Args:
        gesture_img_dir (str): path to LibEMG gestures

    Returns a dictionary mapping the gesture ID to the human-readable gesture name

    """
    with open(gesture_img_dir + "gesture_list.json", "r") as f:
        gid_to_name: dict = json.load(f)
    ret = {}
    for k, v in gid_to_name.items():
        if not isinstance(v, str):
            continue
        ret[int(k)] = v
    return ret


def map_cid_to_name(data_dir: str):
    """
    Map the ODH Class ID (CID) to the human-readable gesture name.

    Params:
        - data_dir: path to the directory where data is stored

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
        class_to_name[val["class_idx"]] = val["class_name"]
    return class_to_name


def map_gid_to_cid(gesture_img_dir: str, data_dir: str):
    """Map LibEMG Gesture ID (GIDs) to ODH Class ID (CIDs)

    Args:
        gesture_img_dir (str): where the images are stored
        data_dir (str): where the data is stored

    Returns a dictionary mapping the gesture ID to the class ID
    """
    gid_to_name = map_gid_to_name(gesture_img_dir)
    name_to_gid = {}
    for g, n in gid_to_name.items():
        if not isinstance(n, str):
            continue
        name_to_gid[n] = g

    cid_to_name = map_cid_to_name(data_dir)
    gid_to_cid = {}
    for c, n in cid_to_name.items():
        gid_to_cid[name_to_gid[n]] = c

    return gid_to_cid


def get_cid_from_gid(gesture_img_dir: str, data_dir: str, gestures: list):
    """Get corresponding CIDs from a list of GIDs. Useful to load data from specific classes from ODH.

    Args:
        gesture_img_dir (str): _description_
        data_dir (str): _description_
        gestures (list): _description_

    Returns a list of class IDs
    """
    g_to_c = map_gid_to_cid(gesture_img_dir, data_dir)
    return [g_to_c[g] for g in gestures]


def get_filter(
    sampling_rate: float,
    bandpass_freqs: float | Iterable = (20, 350),
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
    sampling_rate: float,
    bandpass_freqs: float | list = (20, 450),
    notch_freq: int = 50,
    imu: bool = True,
    **kwargs,
) -> OnlineDataHandler:
    """Get odh and install filters on it. Start listening to the stream.

    Args:
        sampling_rate (float): EMG sampling rate
        bandpass_freqs (float | list, optional). Defaults to [20, 350].
        notch_freq (int, optional). Defaults to 50.
        imu (bool, optional): Use IMU?. Defaults to True.
        kwargs: passed to OnlineDataHandler creator

    Returns:
        OnlineDataHandler: _description_
    """
    odh = OnlineDataHandler(imu_arr=imu, **kwargs)
    odh.install_filter(get_filter(sampling_rate, bandpass_freqs, notch_freq))
    odh.start_listening()
    return odh


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
    train_ui.launch_training(
        odh,
        num_reps,
        rep_time,
        gestures_img_dir,
        output_folder=out_data_dir,
        **kwargs,
    )


if __name__ == "__main__":
    screen_guided_training(
        OnlineDataHandler(), [1, 2, 3, 4, 5], "data/gestures/", 5, 5, "data/train/"
    )
