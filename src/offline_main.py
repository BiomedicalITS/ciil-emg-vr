import numpy as np
from torch import Tensor
import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from libemg.datasets import OneSubjectMyoDataset
from libemg.data_handler import OfflineDataHandler

from models import EmgCNN, TuningEmgCNN

import utils
import globals as g

if __name__ == "__main__":
    dataset = OneSubjectMyoDataset(save_dir="data/", redownload=False)
    odh = dataset.prepare_data(format=OfflineDataHandler)

    # split the dataset into a train, validation, and test set
    # this dataset has a "sets" metadata flag, so lets split
    # train/test using that.
    train_data = odh.isolate_data("sets", [0, 1, 2, 3])
    test_data = odh.isolate_data("sets", [4])
    ft_test_data = odh.isolate_data("sets", [5])

    # lets further split up training and validation based on reps

    fi = utils.get_filter(200, bandpass_freqs=[20, 350], notch_freq=60)
    fi.filter(train_data)
    fi.filter(test_data)
    fi.filter(ft_test_data)

    # for each of these dataset partitions, lets get our windows ready
    ws, wi = 1, 1
    train_windows, train_metadata = train_data.parse_windows(ws, wi)
    test_windows, test_metadata = test_data.parse_windows(ws, wi)
    ft_test_windows, ft_test_metadata = ft_test_data.parse_windows(ws, wi)

    # go from NWC to NCW
    train_windows = train_windows.swapaxes(1, 2)
    test_windows = test_windows.swapaxes(1, 2)
    ft_test_windows = ft_test_windows.swapaxes(1, 2)

    # fix axis for NCHW
    train_windows = np.expand_dims(train_windows, axis=1)
    test_windows = np.expand_dims(test_windows, axis=1)
    ft_test_windows = np.expand_dims(ft_test_windows, axis=1)

    train_windows = utils.process_data(train_windows)
    test_windows = utils.process_data(test_windows)
    ft_test_windows = utils.process_data(ft_test_windows)

    # Create the torch dataloaders both for initial train+test and subsequent fine tuning
    train_loader = DataLoader(
        TensorDataset(Tensor(train_windows), Tensor(train_metadata["classes"])),
        batch_size=32,
        shuffle=True,
    )
    test_loader = DataLoader(
        TensorDataset(Tensor(test_windows), Tensor(test_metadata["classes"])),
        batch_size=256,
    )

    ft_train_loader = DataLoader(
        TensorDataset(Tensor(test_windows), Tensor(test_metadata["classes"])),
        batch_size=32,
        shuffle=True,
    )
    ft_test_loader = DataLoader(
        TensorDataset(Tensor(ft_test_windows), Tensor(ft_test_metadata["classes"])),
        batch_size=256,
    )

    num_classes = len(set(train_metadata["classes"]))

    trainer = L.Trainer(max_epochs=5)
    model = EmgCNN(input_shape=g.EMG_DATA_SHAPE, num_classes=num_classes)

    trainer.fit(model, train_loader)

    print("*" * 60)
    print("Testing original model on test set")
    print("*" * 60)

    print(trainer.test(model, test_loader))
    print(trainer.test(model, ft_test_loader))

    trainer = L.Trainer(max_epochs=5)
    model = utils.get_model(finetune=True, num_classes=num_classes)
    trainer.fit(model, ft_train_loader)

    print("*" * 60)
    print("Testing fine tuning model")
    print("*" * 60)
    print(trainer.test(model, ft_test_loader))
