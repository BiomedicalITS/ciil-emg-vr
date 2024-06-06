import os
from typing import Iterable
import numpy as np
import json
import logging as log
import torch

from libemg.streamers import sifibridge_streamer, myo_streamer, emager_streamer
from libemg.data_handler import OnlineDataHandler
from libemg.filtering import Filter


from models import EmgCNN, EmgSCNN
import globals as g


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


def get_filter(
    sampling_rate: float,
    bandpass_freqs: float | list = [20, 350],
    notch_freq: int = 50,
):
    if not isinstance(bandpass_freqs, Iterable):
        bandpass_freqs = [bandpass_freqs]

    # Create some filters
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


def get_model(
    model_path, emg_shape: tuple, num_classes: int, finetune: bool, scnn=False
):
    """
    Load the most recent model checkpoint and return it
    """
    log.info(f"Loading model from {model_path}")
    chkpt = torch.load(model_path)
    if not scnn:
        n_classes = chkpt["classifier.weight"].shape[0]
        model = EmgCNN(emg_shape, n_classes)
        model.load_state_dict(chkpt)
        model.set_finetune(finetune, num_classes)
    else:
        model = EmgSCNN(emg_shape)
        model.load_state_dict(chkpt)
        embeds = np.load(model_path.replace(".pth", "") + "_embeddings.npy")
        model.set_target_embeddings(embeds)
    return model.eval()


def map_class_to_gestures(data_dir: str):
    """
    g.LIBEMG_GESTURE_IDS define which gestures to download for training.
    ScreenGuidedTraining does not map the gestures to model Class in order
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


def setup_streamer(device: str, notch_freq: int = 50):
    """Setup the streamer for the device

    Args:
        device (str): the sensor (myo, bio, emager)
        notch_freq (int, optional): Notch filter frequency. Defaults to 50.

    Raises:
        ValueError: If the device is not recognized

    Returns:
        process handle
    """
    if device == "myo":
        return myo_streamer(filtered=False, imu=True)
    elif device == "bio":
        return sifibridge_streamer(
            version="1_1",
            emg=True,
            imu=False,
            notch_on=True,
            notch_freq=notch_freq,
        )
    elif device == "emager":
        return emager_streamer()
    else:
        raise ValueError(f"Unknown device: {device}")


def get_online_data_handler(
    sampling_rate: float,
    bandpass_freqs: float | list = [20, 350],
    notch_freq: int = 50,
    imu: bool = True,
    **kwargs,
) -> OnlineDataHandler:
    """_summary_

    Args:
        sampling_rate (float): EMG sampling rate
        bandpass_freqs (float | list, optional). Defaults to [20, 350].
        notch_freq (int, optional). Defaults to 50.
        imu (bool, optional): Use IMU?. Defaults to True.
        kwargs: passed to OnlineDataHandler creator

    Returns:
        OnlineDataHandler: _description_
    """
    if not isinstance(bandpass_freqs, Iterable):
        bandpass_freqs = [bandpass_freqs]
    odh = OnlineDataHandler(imu_arr=imu, **kwargs)
    odh.install_filter(get_filter(sampling_rate, bandpass_freqs, notch_freq))
    odh.start_listening()
    return odh


def process_data(data: np.ndarray):
    data = np.abs(data) / g.EMG_SCALING_FACTOR
    data = moving_average(data, g.EMG_RUNNING_MEAN_LEN)
    return data.astype(np.float32)


def moving_average(x: np.ndarray, N: int):
    orig_shape = x.shape
    x = x.reshape(-1, np.prod(x.shape[1:]))

    for c in range(x.shape[1]):
        x[:, c] = np.convolve(x[:, c], np.ones(N) / N, mode="same")
    return x.reshape(orig_shape)


def tke(data):
    data = np.vstack((data[0:1], data))
    data = np.vstack((data, data[-1:]))
    data = data[1:-1] ** 2 - data[:-2] * data[2:]
    return data


def visualize(device: str, sampling_rate: int, notch_freq: int = 50):
    """
    Visualize the data stream in real time. Takes care of setting up the streamer.
    """
    p = setup_streamer(device, notch_freq)
    odh = get_online_data_handler(sampling_rate, notch_freq=notch_freq, use_imu=False)
    odh.visualize_channels(list(range(8)), 3 * sampling_rate)
    return p


if __name__ == "__main__":
    p = visualize()

    # p = get_most_recent_checkpoint()
    # print(p)
