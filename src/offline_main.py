import numpy as np
from torch import optim, nn, Tensor
import lightning as L
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from libemg.datasets import OneSubjectMyoDataset
from libemg.data_handler import OfflineDataHandler
from libemg.filtering import Filter

from models import EmgCNN, TuningEmgCNN

if __name__ == "__main__":
    dataset = OneSubjectMyoDataset(save_dir="dataset/", redownload=False)
    odh = dataset.prepare_data(format=OfflineDataHandler)

    # split the dataset into a train, validation, and test set
    # this dataset has a "sets" metadata flag, so lets split
    # train/test using that.
    not_test_data = odh.isolate_data("sets", [0, 1, 2, 3, 4])
    test_data = odh.isolate_data("sets", [5])
    # lets further split up training and validation based on reps
    train_data = not_test_data.isolate_data("sets", [0, 1, 2, 3])
    valid_data = not_test_data.isolate_data("sets", [4])

    # let's perform the filtering on the dataset too (neural networks like
    # inputs that are standardized).
    fi = Filter(sampling_frequency=200)
    fi.install_filters({"name": "standardize", "data": train_data})
    fi.install_filters({"name": "highpass", "cutoff": 20, "order": 2})
    # fi.install_filters({"name": "notch", "cutoff": 50, "bandwidth": 3})
    fi.install_filters({"name": "notch", "cutoff": 60, "bandwidth": 3})
    fi.filter(train_data)
    fi.filter(valid_data)
    fi.filter(test_data)

    # for each of these dataset partitions, lets get our windows ready
    ws, wi = 1, 1
    train_windows, train_metadata = train_data.parse_windows(ws, wi)
    valid_windows, valid_metadata = valid_data.parse_windows(ws, wi)
    test_windows, test_metadata = test_data.parse_windows(ws, wi)

    # go from NWC to NCW
    train_windows = train_windows.swapaxes(1, 2)
    valid_windows = valid_windows.swapaxes(1, 2)
    test_windows = test_windows.swapaxes(1, 2)

    # fix axis for NCHW
    train_windows = np.expand_dims(train_windows, axis=1)
    valid_windows = np.expand_dims(valid_windows, axis=1)
    test_windows = np.expand_dims(test_windows, axis=1)

    # Create the torch datasets
    train_loader = DataLoader(
        TensorDataset(Tensor(train_windows), Tensor(train_metadata["classes"])),
        batch_size=32,
        shuffle=True,
    )
    valid_loader = DataLoader(
        TensorDataset(Tensor(valid_windows), Tensor(valid_metadata["classes"])),
        batch_size=256,
    )
    test_loader = DataLoader(
        TensorDataset(Tensor(test_windows), Tensor(test_metadata["classes"])),
        batch_size=256,
    )

    # model = EmgCNN(input_shape=(8, 1), num_classes=5)
    model = EmgCNN.load_from_checkpoint(
        checkpoint_path="lightning_logs/version_7/checkpoints/epoch=4-step=3770.ckpt",
        input_shape=(8, 1),
        num_classes=5,
    )
    model = TuningEmgCNN(model, num_classes=5)

    trainer = L.Trainer(max_epochs=5)
    trainer.fit(model, train_loader, valid_loader)
    print(trainer.test(model, test_loader))

    print("Done!")
