import os
import shutil
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score
import matplotlib.pyplot as plt


import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from libemg.screen_guided_training import ScreenGuidedTraining
from libemg.utils import make_regex
from libemg.data_handler import OfflineDataHandler, OnlineDataHandler

from models import EmgSCNN
import emager_py.data_processing as dp
import emager_py.torch.utils as etu

import utils


def get_training_data(
    online_data_handler: OnlineDataHandler,
    gestures: list,
    num_reps: int,
    rep_time: int,
    output_dir: str,
    **kwargs,
):
    """
    Params:
        - gestures: list of gesture ids
        - num_reps: number of repetitions per gesture
        - rep_time: time in seconds for each repetition
        - kwargs: additional parameters for the training UI, passed as key-values to `ScreenGuidedTraining.launch_training`
    """
    gestures_dir = g.LIBEMG_GESTURES_DIR
    shutil.rmtree(gestures_dir, ignore_errors=True)
    train_ui = ScreenGuidedTraining()
    train_ui.download_gestures(gestures, gestures_dir)
    train_ui.launch_training(
        online_data_handler,
        num_reps,
        rep_time,
        gestures_dir,
        output_folder=output_dir,
        **kwargs,
    )


def get_offline_datahandler(repetitions: list, data_dir: str):
    classes_values = [str(gid) for gid in range(len(g.LIBEMG_GESTURE_IDS))]
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


def prepare_triplets(odh: OfflineDataHandler, ws, wi, batch_size, shuffle):
    windows, meta = odh.parse_windows(ws, wi)
    windows = windows.swapaxes(1, 2)
    windows = np.expand_dims(windows, axis=1)
    windows = utils.process_data(windows)
    labels = meta["classes"]

    anchor, positive, negative = dp.generate_triplets(windows, labels, 2000)

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


def prepare_data(odh: OfflineDataHandler, ws, wi, batch_size, shuffle):
    windows, meta = odh.parse_windows(ws, wi)
    windows = windows.swapaxes(1, 2)
    windows = np.expand_dims(windows, axis=1)
    windows = utils.process_data(windows)
    cutoff = len(windows) - len(windows) % batch_size
    windows = windows[:cutoff]
    labels = meta["classes"][:cutoff]
    dataloader = DataLoader(
        TensorDataset(torch.from_numpy(windows), torch.from_numpy(labels)),
        batch_size=batch_size,
        shuffle=shuffle,
    )
    return dataloader


def train_model(model: EmgSCNN, data_dir: str, train_reps: list, test_reps: list):
    if not isinstance(train_reps, list):
        train_reps = [train_reps]
    if not isinstance(test_reps, list):
        test_reps = [test_reps]

    odh = get_offline_datahandler(train_reps + test_reps, data_dir)

    train_data = odh.isolate_data("reps", train_reps)
    test_data = odh.isolate_data("reps", test_reps)

    train_loader = prepare_triplets(train_data, 1, 1, 64, True)
    model = model.train()
    trainer = L.Trainer(
        max_epochs=10,
        callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)],
    )
    trainer.fit(model, train_loader)
    targets = None
    if len(test_reps) > 0:
        test_loader = prepare_data(test_data, 1, 1, 128, False)
        embeddings, labels = etu.get_all_embeddings(model, test_loader, "cpu")
        targets = dp.get_n_shot_embeddings(embeddings, labels, len(set(labels)), -1)
        model.set_target_embeddings(targets)
        trainer.test(model, test_loader)

    return model, targets


def test_model(model: EmgSCNN, data_dir: str, test_reps: list):
    """Test model. Returns (y_pred, y_true)."""
    if not isinstance(test_reps, list):
        test_reps = [test_reps]

    model.eval()
    odh = get_offline_datahandler(test_reps, data_dir)
    test_loader = prepare_data(odh, 1, 1, 128, False)
    embeddings, labels = etu.get_all_embeddings(model, test_loader, "cpu")
    targets = dp.get_n_shot_embeddings(embeddings, labels, len(set(labels)), -1)
    model.set_target_embeddings(targets)
    y_pred, y_true = [], []
    for i, batch in enumerate(test_loader):
        ret = model.test_step(batch, i)
        y_pred.extend(ret["y_pred"])
        y_true.extend(ret["y_true"])
    return y_pred, y_true


def get_reps(path: str):
    """
    Get all repetitions from a directory R_*
    """
    return list(set([int(f.split("_")[1]) for f in os.listdir(path) if "R_" in f]))


def main(
    sample_data,
    finetune,
    data_dir,
    device: str,
    emg_shape: tuple,
    emg_fs: int,
    emg_notch_freq: int,
    gestures_list: list,
    model_out_path: str,
):
    if sample_data:
        utils.setup_streamer(device, emg_notch_freq)
        odh = utils.get_online_data_handler(
            emg_fs, notch_freq=emg_notch_freq, imu=False
        )
        get_training_data(odh, gestures_list, 1 if finetune else 5, 5, data_dir)

    reps = get_reps(data_dir)
    if len(reps) == 1:
        train_reps = reps
        test_reps = []
    else:
        train_reps = reps[: int(0.8 * len(reps))]
        test_reps = reps[int(0.8 * len(reps)) :]

    print("Training on reps:", train_reps)
    print("Testing on reps:", test_reps)

    model = (
        EmgSCNN(emg_shape)
        if not finetune
        else utils.get_model(
            model_out_path,
            emg_shape,
            num_classes=len(gestures_list),
            finetune=True,
            scnn=True,
        )
    )
    model, embeddings = train_model(model, data_dir, train_reps, test_reps)
    torch.save(model.state_dict(), model_out_path)
    if embeddings is not None:
        np.save(model_out_path.replace(".pth", "") + "_embeddings.npy", embeddings)


if __name__ == "__main__":
    import globals as g

    SAMPLE_DATA = False

    # y_true, y_pred = test_model(
    #     utils.get_model(
    #         g.MODEL_PATH, g.EMG_DATA_SHAPE, len(g.LIBEMG_GESTURE_IDS), False, True
    #     ),
    #     g.TRAIN_DATA_DIR,
    #     get_reps(g.TRAIN_DATA_DIR)[-1],
    # )

    # ConfusionMatrixDisplay.from_predictions(y_true, y_pred, normalize="true")
    # print("Raw accuracy: ", accuracy_score(y_true, y_pred, normalize=True))
    # print(utils.map_class_to_gestures(g.TRAIN_DATA_DIR))
    # plt.show()
    # exit()

    main(
        sample_data=SAMPLE_DATA,
        finetune=False,
        data_dir=g.TRAIN_DATA_DIR,
        device=g.DEVICE,
        emg_shape=g.EMG_DATA_SHAPE,
        emg_fs=g.EMG_SAMPLING_RATE,
        emg_notch_freq=g.EMG_NOTCH_FREQ,
        gestures_list=g.LIBEMG_GESTURE_IDS,
        model_out_path=g.MODEL_PATH,
    )
