import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from libemg.data_handler import OfflineDataHandler
from libemg.utils import make_regex

from emager_py import data_processing as dp

from nfc_emg.sensors import EmgSensor


def process_data(data: np.ndarray, sensor: EmgSensor):
    """Process EMG data.

    Args:
        data (np.ndarray): EMG data with shape (n_samples, 1, *emg_shape)
        device (EmgSensor): The device used

    Returns:
        Processed data with shape (n_samples, 1, *emg_shape)
    """
    data = np.abs(sensor.reorder(data)) * sensor.emg_factor
    data = moving_average(data, sensor.moving_avg_n)
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


def get_offline_datahandler(
    data_dir: str,
    classes: list,
    repetitions: list,
):
    """
    Get data handler from a pre-recorded dataset.

    Params:
        - data_dir: directory where data is stored
        - classes: Class IDs to load into odh
        - repetitions: list of repetitions to load into odh
    """

    classes_values = [str(c) for c in classes]
    classes_regex = make_regex(
        left_bound="_C_", right_bound="_EMG.csv", values=classes_values
    )

    reps_values = [str(rep) for rep in repetitions]
    reps_regex = make_regex(left_bound="R_", right_bound="_C_", values=reps_values)

    dic = {
        "reps": reps_values,
        "reps_regex": reps_regex,
        "classes": classes_values,
        "classes_regex": classes_regex,
    }

    odh = OfflineDataHandler()
    odh.get_data(folder_location=data_dir, filename_dic=dic, delimiter=",")
    return odh


def prepare_data(odh: OfflineDataHandler, sensor: EmgSensor, ws, wi):
    """Prepare data stored in an OfflineDataHandler for training.

    Returns:
        tuple of np.ndarray with shapes (n_samples, 1, *emg_shape), (n_samples,)
    """
    windows, meta = odh.parse_windows(ws, wi)
    windows = windows.swapaxes(1, 2)
    windows = np.expand_dims(windows, axis=1)
    windows = process_data(windows, sensor)
    labels = meta["classes"]
    return windows, labels


def get_triplet_dataloader(
    odh: OfflineDataHandler,
    sensor: EmgSensor,
    ws: int,
    wi: int,
    batch_size: int,
    shuffle: bool,
    n_triplets,
):
    """
    Get a triplet dataloader.

    Params:
        - odh: offline data handler
        - sensor: emg sensor
        - ws: window size
        - wi: window increment
        - batch_size: batch size
        - shuffle: shuffle data

    Returns a dataloader which yields (anchor, positive, negative) batches.
    """
    windows, labels = prepare_data(odh, sensor, ws, wi)
    anchor, positive, negative = dp.generate_triplets(windows, labels, n_triplets)
    dataloader = DataLoader(
        TensorDataset(
            torch.from_numpy(anchor),
            torch.from_numpy(positive),
            torch.from_numpy(negative),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataloader


def get_dataloader(
    odh: OfflineDataHandler,
    sensor: EmgSensor,
    ws: int,
    wi: int,
    batch_size: int,
    shuffle: bool,
):
    """
    Get a dataloader for training.

    Params:
        - odh: offline data handler
        - sensor: emg sensor
        - ws: window size
        - wi: window increment
        - batch_size: batch size
        - shuffle: shuffle data

    Returns a dataloader which yields (windows, labels) batches.
    """
    windows, labels = prepare_data(odh, sensor, ws, wi)
    # cutoff = len(windows) - len(windows) % batch_size
    # windows = windows[:cutoff]
    # labels = labels[:cutoff]
    dataloader = DataLoader(
        TensorDataset(torch.from_numpy(windows), torch.from_numpy(labels)),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataloader
