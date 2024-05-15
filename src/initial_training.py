import numpy as np
from torch import Tensor
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

import lightning as L

# from lightning.pytorch.callbacks import EarlyStopping

from libemg.screen_guided_training import ScreenGuidedTraining
from libemg.utils import make_regex
from libemg.data_handler import OfflineDataHandler, OnlineDataHandler

import models
import utils
import globals as g


def get_training_data(
    online_data_handler: OnlineDataHandler,
    gestures,
    num_reps,
    rep_time,
    output_dir,
    **kwargs
):
    """
    Params:
        - gestures: list of gesture ids
        - num_reps: number of repetitions per gesture
        - rep_time: time in seconds for each repetition
        - kwargs: additional parameters for the training UI, passed as key-values to `ScreenGuidedTraining.launch_training`
    """
    gestures_dir = "data/gestures/"
    train_ui = ScreenGuidedTraining()
    train_ui.download_gestures(gestures, gestures_dir)
    train_ui.launch_training(
        online_data_handler,
        num_reps,
        rep_time,
        gestures_dir,
        output_folder=output_dir,
        **kwargs
    )


def train_model(data_dir: str, train_reps: list, test_reps: list):
    """
    TODO: validation reps?
    """

    if not isinstance(train_reps, list):
        train_reps = [train_reps]
    if not isinstance(test_reps, list):
        test_reps = [test_reps]

    classes_values = [str(gid) for gid in range(len(g.LIBEMG_GESTURE_IDS))]
    classes_regex = make_regex(
        left_bound="_C_", right_bound="_EMG.csv", values=classes_values
    )

    reps_values = [str(rep) for rep in train_reps + test_reps]
    reps_regex = make_regex(left_bound="R_", right_bound="_C_", values=reps_values)

    dic = {
        "reps": reps_values,
        "reps_regex": reps_regex,
        "classes": classes_values,
        "classes_regex": classes_regex,
    }

    odh = OfflineDataHandler()
    odh.get_data(folder_location=data_dir, filename_dic=dic, delimiter=",")

    train_data = odh.isolate_data("reps", train_reps)
    test_data = odh.isolate_data("reps", test_reps)

    # 1-sample window and 1-sample stride
    ws, wi = 1, 1
    train_windows, train_metadata = train_data.parse_windows(ws, wi)
    test_windows, test_metadata = test_data.parse_windows(ws, wi)

    # go from NWC to NCW
    train_windows = train_windows.swapaxes(1, 2)
    test_windows = test_windows.swapaxes(1, 2)

    # add axis for NCHW
    train_windows = np.expand_dims(train_windows, axis=1)
    test_windows = np.expand_dims(test_windows, axis=1)

    # Process the data
    train_windows = utils.process_data(train_windows)
    test_windows = utils.process_data(test_windows)

    # Create the torch datasets
    train_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(train_windows), torch.from_numpy(train_metadata["classes"])
        ),
        batch_size=32,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(
            torch.from_numpy(test_windows), torch.from_numpy(test_metadata["classes"])
        ),
        batch_size=128,
    )

    model = models.EmgCNN(
        input_shape=test_windows.shape[-2:],
        num_classes=len(np.unique(train_metadata["classes"])),
    )

    trainer = L.Trainer(max_epochs=10)
    trainer.fit(model, train_loader)
    trainer.test(model, test_loader)
    return model


if __name__ == "__main__":
    utils.setup_streamer()
    odh = utils.get_online_data_handler(
        g.EMG_SAMPLING_RATE, notch_freq=g.EMG_NOTCH_FREQ, use_imu=g.USE_IMU
    )
    get_training_data(odh, g.LIBEMG_GESTURE_IDS, 2, 3, g.OFFLINE_DATA_DIR)
    train_model(g.OFFLINE_DATA_DIR, 0, 1)
