import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from libemg.data_handler import OfflineDataHandler
from libemg.utils import make_regex

from emager_py import data_processing as dp

from nfc_emg.sensors import EmgSensor, EmgSensorType


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


def prepare_data(odh: OfflineDataHandler, sensor: EmgSensor):
    """
    Prepare data stored in an OfflineDataHandler.

    Returns:
        tuple of np.ndarray with shapes (N, C, W), (N,), where C is the # channels and W is window size
    """
    # windows has shape (N, C, W)
    windows, meta = odh.parse_windows(sensor.window_size, sensor.window_increment)
    labels = meta["classes"]
    return windows, labels


def get_triplet_dataloader(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
    n_triplets,
):
    """
    Get a triplet dataloader.

    Params:
        - data: (N, H, W)
        - labels: (N,)
        - batch_size: batch size
        - shuffle: shuffle data

    Returns a dataloader which yields (anchor, positive, negative) batches.
    """
    anchor, positive, negative = dp.generate_triplets(
        data, labels, n_triplets - n_triplets % batch_size
    )
    dataloader = DataLoader(
        TensorDataset(
            torch.from_numpy(anchor[:, np.newaxis, ...]),
            torch.from_numpy(positive[:, np.newaxis, ...]),
            torch.from_numpy(negative[:, np.newaxis, ...]),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataloader


def get_dataloader(
    data: np.ndarray,
    labels: np.ndarray,
    batch_size: int,
    shuffle: bool,
):
    """
    Get a dataloader for training.

    Params:
        - data: (N, H, W)
        - labels: (N,)
        - batch_size: batch size
        - shuffle: shuffle data

    Returns a dataloader which yields (windows, labels) batches.
    """
    cutoff = len(labels) - len(labels) % batch_size
    data = data[:cutoff]
    labels = labels[:cutoff]
    dataloader = DataLoader(
        TensorDataset(
            torch.from_numpy(data[:, np.newaxis, ...]),
            torch.from_numpy(labels),
        ),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataloader
