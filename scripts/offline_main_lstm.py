import numpy as np
from tqdm import tqdm

from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from nfc_emg import datasets
from nfc_emg import utils
from nfc_emg.sensors import EmgSensor
from nfc_emg.paths import NfcPaths

import configs as g


class EmgLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes):
        super().__init__()

        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        self.fc1 = nn.Linear(hidden_size, 128)
        self.bn1 = nn.BatchNorm1d(hidden_size)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        # x.shape = (N, W, C)

        out, _ = self.lstm(x)

        out = F.dropout(F.relu(self.bn1(self.fc1(out[:, -1, :]))))
        out = F.dropout(F.relu(self.bn2(self.fc2(out))))
        out = self.out(out)
        return out


class EmgCnnLstm(nn.Module):
    def __init__(self, input_channels, lstm_hidden_size, num_classes):
        super().__init__()

        # CNN Layers
        self.conv1 = nn.Conv1d(
            input_channels,
            32,
            kernel_size=3,
            stride=1,
            padding="same",
            padding_mode="circular",
        )
        self.bnc1 = nn.BatchNorm1d(32)
        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bnc2 = nn.BatchNorm1d(64)

        # LSTM Layer
        self.lstm = nn.LSTM(8, lstm_hidden_size, num_layers=2, batch_first=True)

        # Fully Connected Layer
        self.fc1 = nn.Linear(lstm_hidden_size, 128)
        self.bn1 = nn.BatchNorm1d(lstm_hidden_size)

        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)

        self.out = nn.Linear(64, num_classes)

    def forward(self, x):
        # x = (batch, seq_length, input_channels)

        # CNN
        x = F.relu(self.bnc1(self.conv1(x)))
        x = F.relu(self.bnc2(self.conv2(x)))

        # LSTM
        out, _ = self.lstm(x)

        # Fully Connected Layer (use last time step)
        out = F.dropout(F.relu(self.bn1(self.fc1(out[:, -1, :]))))
        out = F.dropout(F.relu(self.bn2(self.fc2(out))))
        out = self.out(out)
        return out


def print_model_size(model):
    total_params = sum(p.numel() for p in model.parameters())
    total_memory = total_params * 4  # Each parameter is 4 bytes (float32)
    total_memory_mb = total_memory / (1024**2)  # Convert to MB
    print(f"Model Parameters: {total_params:,}")
    print(f"Model Size: {total_memory_mb:.2f} MB")


def __main():
    SAMPLE_DATA = False
    FINETUNE = False

    # SAMPLE_DATA = True
    # FINETUNE = True

    sensor = EmgSensor(
        g.SENSOR, window_size_ms=50, window_inc_ms=25, majority_vote_ms=0
    )

    # paths = NfcPaths(f"data/{sensor.get_name()}", -1)
    paths = NfcPaths(f"data/0/{sensor.get_name()}", "adap")
    paths.gestures = "data/gestures/"
    paths.test = "pre_test/"

    train_dir = paths.get_train()
    test_dir = paths.get_test()

    train_odh = datasets.get_offline_datahandler(
        train_dir, range(len(g.FUNCTIONAL_SET)), range(5)
    )
    val_odh = datasets.get_offline_datahandler(
        test_dir, range(len(g.FUNCTIONAL_SET)), range(2)
    )

    train_win, train_labels = datasets.prepare_data(train_odh, sensor)
    val_win, val_labels = datasets.prepare_data(val_odh, sensor)

    train_win = np.abs(train_win)
    val_win = np.abs(val_win)

    train_win = (train_win - np.mean(train_win, axis=0, keepdims=True)) / np.std(
        train_win, axis=0, keepdims=True
    )
    val_win = (val_win - np.mean(val_win, axis=0, keepdims=True)) / np.std(
        val_win, axis=0, keepdims=True
    )

    # (N, C, W) -> (N, W, C)
    train_win = np.swapaxes(train_win, 1, 2)
    val_win = np.swapaxes(val_win, 1, 2)

    X_train = torch.tensor(train_win).float()
    y_train = torch.tensor(train_labels).long()
    X_test = torch.tensor(val_win).float()
    y_test = torch.tensor(val_labels).long()

    # model = EmgLSTM(np.prod(sensor.emg_shape), 128, 2, len(g.FUNCTIONAL_SET)).to("cuda")
    model = EmgCnnLstm(X_train.shape[1], 128, len(g.FUNCTIONAL_SET)).to("cuda")

    print_model_size(model)

    optimizer = optim.AdamW(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()
    loader = data.DataLoader(
        data.TensorDataset(X_train, y_train),
        batch_size=64,
        shuffle=True,
    )

    n_epochs = 30
    for epoch in range(n_epochs):
        model.train()
        for X_batch, y_batch in tqdm(loader, leave=False):
            y_pred = model(X_batch.to("cuda")).cpu()
            loss = loss_fn(y_pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # Validation
        if epoch != n_epochs - 1 and epoch % 10 != 0:
            continue
        model.eval()
        with torch.no_grad():
            y_pred = model(X_train.to("cuda")).cpu()
            train_loss = loss_fn(y_pred, y_train)
            train_acc = (y_pred.argmax(1) == y_train).float().mean()

            y_pred = model(X_test.to("cuda")).cpu()
            test_loss = loss_fn(y_pred, y_test)
            test_acc = (y_pred.argmax(1) == y_test).float().mean()
        print(
            "Epoch %d: train loss %.4f, acc %.4f, test loss %.4f, acc %.4f"
            % (epoch, train_loss, 100 * train_acc, test_loss, 100 * test_acc)
        )

    with torch.no_grad():
        y_pred_labels = model(X_test.to("cuda")).cpu().argmax(1)

    # Compute confusion matrix
    cm = confusion_matrix(y_test.numpy(), y_pred_labels.numpy())
    cm_normalized = cm.astype("float") / cm.sum(axis=1, keepdims=True)

    test_gesture_names = utils.get_name_from_gid(
        paths.gestures, train_dir, g.FUNCTIONAL_SET
    )

    # Display the confusion matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm_normalized, display_labels=test_gesture_names
    )
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


if __name__ == "__main__":
    __main()
