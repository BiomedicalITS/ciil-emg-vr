import numpy as np
import logging as log

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from typing import Iterable

from libemg.offline_metrics import OfflineMetrics
from libemg.emg_classifier import EMGClassifier
from libemg.feature_extractor import FeatureExtractor

from emager_py.majority_vote import majority_vote

from nfc_emg import datasets, utils
from nfc_emg.sensors import EmgSensor


class EmgCNN(L.LightningModule):
    def __init__(self, num_channels: int, emg_shape: tuple, num_classes: int):
        """
        Parameters:
            - num_channels: number of channels (eg number of different features)
            - emg_shape: shape of EMG. Must be a tuple, eg (8,) for Armbands and (4, 16) for Emager
            - num_classes: number of classes
        """
        super().__init__()
        self.save_hyperparameters()

        self.scaler = StandardScaler()

        if len(emg_shape) == 1:
            ConvNd = nn.Conv1d
            BatchNormNd = nn.BatchNorm1d
        else:
            ConvNd = nn.Conv2d
            BatchNormNd = nn.BatchNorm2d

        self.emg_shape = emg_shape
        self.num_channels = num_channels

        conv_sizes = [num_channels, 32, 32]
        fc_sizes = [conv_sizes[-1] * np.prod(self.emg_shape), 256]

        log.info(f"Conv layer channels: {conv_sizes}")
        log.info(f"FC layer neurons: {fc_sizes}")

        layers = []
        for i in range(1, len(conv_sizes)):
            convlen = 5 if i == 1 else 3
            # convlen = 3
            layers.append(
                ConvNd(
                    conv_sizes[i - 1],
                    conv_sizes[i],
                    convlen,
                    padding="same",
                    # padding_mode="circular",
                )
            )
            layers.append(BatchNormNd(conv_sizes[i]))
            layers.append(nn.LeakyReLU())
        layers.append(nn.Flatten())
        for i in range(1, len(fc_sizes)):
            layers.append(nn.Linear(fc_sizes[i - 1], fc_sizes[i]))
            layers.append(nn.BatchNorm1d(fc_sizes[i]))
            layers.append(nn.Dropout(0.2))
            layers.append(nn.LeakyReLU())

        self.feature_extractor = nn.Sequential(*layers)
        self.classifier = nn.Linear(fc_sizes[-1], num_classes)

    def forward(self, x):
        x = torch.reshape(x, (-1, self.num_channels, *self.emg_shape))
        out = self.feature_extractor(x)
        logits = self.classifier(out)
        return logits

    def convert_input(self, x):
        """
        Pre-process the input by scaling it, reshaping and converting it to a Torch tensor
        """
        x = self.scaler.transform(x)
        x = x.reshape(-1, self.num_channels, *self.emg_shape)

        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return x.float().to(self.device)

    # ----- Lightning -----

    def training_step(self, batch, batch_idx, logging=True):
        x, y_true = batch
        y = self(x)
        loss = F.cross_entropy(y, y_true)

        if len(y_true.shape) == 2:
            # class probabilities
            y_true = torch.argmax(y_true, dim=1)

        y = np.argmax(y.cpu().detach().numpy(), axis=1)
        y_true = y_true.cpu().detach().numpy()

        acc = accuracy_score(y_true, y, normalize=True)

        if logging:
            self.log("train_loss", loss)
            self.log("train_acc", acc)

        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = F.cross_entropy(y, y_true)

        if len(y_true.shape) == 2:
            # class probabilities
            y_true = torch.argmax(y_true, dim=1)

        y = np.argmax(y.cpu().detach().numpy(), axis=1)
        y_true = y_true.cpu().detach().numpy()

        acc = accuracy_score(y_true, y, normalize=True)

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def test_step(self, batch, batch_idx, logging=True):
        x, y_true = batch
        y = self(x)
        loss: float = F.cross_entropy(y, y_true)

        if len(y_true.shape) == 2:
            # class probabilities
            y_true = torch.argmax(y_true, dim=1)

        y = np.argmax(y.cpu().detach().numpy(), axis=1)
        y_true = y_true.cpu().detach().numpy()

        acc = accuracy_score(y_true, y, normalize=True)

        if logging:
            self.log("test_loss", loss)
            self.log("test_acc", acc)

        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    # ----- LibEMG -----

    def predict_proba(self, x) -> np.ndarray:
        x = self.convert_input(x)
        with torch.no_grad():
            return F.softmax(self(x), dim=1).cpu().detach().numpy()

    def predict(self, x) -> np.ndarray:
        return np.argmax(self.predict_proba(x), axis=1)

    def fit(self, data: np.ndarray, labels: np.ndarray):
        """
        Fit the model from data. Data should consist of the extracted features (N, C, L), where L is the number of EMG channels.

        This function takes care of data normalization.
        """
        # train_data, val_data, train_labels, val_labels = train_test_split(
        #     data, labels, test_size=0.2
        # )
        train_data, train_labels = data, labels

        train_data = self.convert_input(train_data)
        train_labels = torch.from_numpy(train_labels).to(self.device)
        train_dl = DataLoader(TensorDataset(train_data, train_labels), batch_size=32)

        # val_data = self.convert_input(val_data)
        # val_labels = torch.from_numpy(val_labels).to(self.device)
        # val_dl = DataLoader(TensorDataset(val_data, val_labels), batch_size=100)

        self.train()
        optim = torch.optim.AdamW(self.parameters(), lr=1e-3)
        # for _ in range(5):
        loss_avg = 0
        for i, batch in enumerate(train_dl):
            if len(batch[0]) == 1:
                continue
            optim.zero_grad()
            loss = self.training_step(batch, i, False)
            loss.backward()
            optim.step()
            loss_avg += loss.item()
        self.eval()
        return {"loss": loss_avg, "acc": 0.0}
        # rets = {}
        # num_batches = 0
        # for i, batch in enumerate(val_dl):
        #     with torch.no_grad():
        #         ret = self.test_step(batch, i, False)
        #     num_batches += 1
        #     if rets == {}:
        #         rets = ret
        #     else:
        #         rets = {k: v + ret[k] for k, v in rets.items()}
        # return {k: v / num_batches for k, v in rets.items()}


class EmgMLP(L.LightningModule):
    def __init__(self, num_features, num_classes):
        """
        Parameters:
            - input_shape: shape of input data
            - num_classes: number of classes
        """
        super().__init__()
        self.save_hyperparameters()

        self.scaler = StandardScaler()

        hl_sizes = [num_features, 128, 256]
        # hl_sizes = [num_features, 100]

        net = [
            nn.Flatten(),
        ]

        for i in range(len(hl_sizes) - 1):
            net.append(nn.Linear(hl_sizes[i], hl_sizes[i + 1]))
            net.append(nn.BatchNorm1d(hl_sizes[i + 1]))
            net.append(nn.LeakyReLU())
            net.append(nn.Dropout(0.2))

        self.feature_extractor = nn.Sequential(*net)
        self.classifier = nn.Linear(hl_sizes[-1], num_classes)

    def forward(self, x):
        out = self.feature_extractor(x)
        logits = self.classifier(out)
        return logits

    def convert_input(self, x):
        """Convert arbitrary input to a Torch tensor

        Args:
            x (_type_): _description_

        Returns:
            _type_: _description_
        """
        x = self.scaler.transform(x)
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x)
        return x.float().to(self.device)

    # ----- Lightning -----

    def training_step(self, batch, batch_idx, logging=True):
        x, y_true = batch
        y = self(x)
        loss = F.cross_entropy(y, y_true)

        if len(y_true.shape) == 2:
            # class probabilities
            y_true = torch.argmax(y_true, dim=1)

        y = np.argmax(y.cpu().detach().numpy(), axis=1)
        y_true = y_true.cpu().detach().numpy()

        acc = accuracy_score(y_true, y, normalize=True)

        if logging:
            self.log("train_loss", loss)
            self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = F.cross_entropy(y, y_true)

        if len(y_true.shape) == 2:
            # class probabilities
            y_true = torch.argmax(y_true, dim=1)

        y = np.argmax(y.cpu().detach().numpy(), axis=1)
        y_true = y_true.cpu().detach().numpy()

        acc = accuracy_score(y_true, y, normalize=True)

        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def test_step(self, batch, batch_idx, logging=True):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y: torch.Tensor = self(x)
        loss = F.cross_entropy(y, y_true)

        if len(y_true.shape) == 2:
            # class probabilities
            y_true = torch.argmax(y_true, dim=1)

        y = np.argmax(y.cpu().detach().numpy(), axis=1)
        y_true = y_true.cpu().detach().numpy()

        acc = accuracy_score(y_true, y, normalize=True)
        if logging:
            self.log("test_loss", loss)
            self.log("test_acc", acc)
        return {"loss": loss, "acc": acc}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    # ----- LibEMG -----

    def predict_proba(self, x):
        x = self.convert_input(x)
        with torch.no_grad():
            return F.softmax(self(x), dim=1).cpu().detach().numpy()

    def predict(self, x):
        return np.argmax(self.predict_proba(x), axis=1)

    def fit(self, data, labels):
        train_data, val_data, train_labels, val_labels = train_test_split(
            data, labels, test_size=0.2
        )

        train_data = self.convert_input(train_data)
        train_labels = torch.from_numpy(train_labels).to(self.device)
        train_dl = DataLoader(TensorDataset(train_data, train_labels), batch_size=32)

        val_data = self.convert_input(val_data)
        val_labels = torch.from_numpy(val_labels).to(self.device)
        val_dl = DataLoader(TensorDataset(val_data, val_labels), batch_size=100)

        optim = self.configure_optimizers()
        self.train()
        for i, batch in enumerate(train_dl):
            if len(batch[0]) < 2:
                continue
            loss = self.training_step(batch, i, False)

            loss.backward()
            optim.step()
            optim.zero_grad()

        self.eval()
        rets = {}
        num_batches = 0
        for i, batch in enumerate(val_dl):
            ret = self.test_step(batch, i, False)
            num_batches += 1
            if rets == {}:
                rets = ret
            else:
                rets = {k: v + ret[k] for k, v in rets.items()}
        return {k: v / num_batches for k, v in rets.items()}


class EmgSCNN(L.LightningModule):
    def __init__(self, input_shape: tuple):
        """
        Parameters:
            - input_shape: EMG input shape (H, W)
        """
        super().__init__()
        self.save_hyperparameters()

        hl_sizes = [32, 32, 32, 128, 64]

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, hl_sizes[0], 5, padding=2),
            nn.BatchNorm2d(hl_sizes[0]),
            nn.LeakyReLU(),
            nn.Conv2d(hl_sizes[0], hl_sizes[1], 3, padding=1),
            nn.BatchNorm2d(hl_sizes[1]),
            nn.LeakyReLU(),
            nn.Conv2d(hl_sizes[1], hl_sizes[2], 3, padding=1),
            nn.BatchNorm2d(hl_sizes[2]),
            nn.LeakyReLU(),
            nn.Flatten(),
            nn.Linear(hl_sizes[2] * np.prod(input_shape), hl_sizes[3]),
            nn.BatchNorm1d(hl_sizes[3]),
            nn.LeakyReLU(),
            nn.Dropout(),
            nn.Linear(hl_sizes[3], hl_sizes[4]),
            nn.BatchNorm1d(hl_sizes[4]),
        )

    def forward(self, x):
        return self.feature_extractor(x)

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x1, x2, x3 = batch
        anchor, positive, negative = self(x1), self(x2), self(x3)
        loss = F.triplet_margin_loss(anchor, positive, negative, margin=0.2)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x1, x2, x3 = batch
        anchor, positive, negative = self(x1), self(x2), self(x3)
        loss = F.triplet_margin_loss(anchor, positive, negative, margin=0.2)
        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


class CosineSimilarity(BaseEstimator):
    def __init__(self, num_classes: int | None = None, dims: int | None = None):
        """
        Create a cosine similarity classifier.
        """
        super().__init__()

        if num_classes is not None:
            self.features = np.zeros((num_classes, dims))
            self.n_samples = np.zeros(num_classes)
        else:
            self.features = None
            self.n_samples = None

    def __cosine_similarity(self, X, labels: bool):
        dists = cosine_similarity(X, self.features)
        if labels:
            return np.argmax(dists, axis=1)
        return dists

    def fit(self, X, y):
        """Fit the similarity classifier.

        Args:
            X : the features of shape (n_samples, n_features)
            y: the labels of shape (n_samples,)
        """
        if self.features is None:
            self.features = np.zeros((len(np.unique(y)), X.shape[1]))
            self.n_samples = np.zeros(len(np.unique(y)))

        tmp_features = self.features
        for i in range(len(self.features)):
            tmp_features[i] *= self.n_samples[i]

        for c in np.unique(y):
            c_labels = y == c
            self.n_samples[c] += len(c_labels)
            tmp_features[c] += np.sum(X[c_labels], axis=0)

        for i in range(len(self.features)):
            if self.n_samples[i] == 0:
                continue
            self.features = tmp_features / self.n_samples[i]

    def predict(self, X):
        return self.__cosine_similarity(X, True)

    def predict_proba(self, X):
        dists = self.__cosine_similarity(X, False)
        return (dists + 1) / 2.0  # scale [-1, 1] to [0, 1]


class EmgSCNNWrapper:
    def __init__(
        self,
        model: EmgSCNN,
        classifier: BaseEstimator,
    ):
        """The SCNN model wrapper. It includes an EMGSCNN model and a classifier."""
        self.model = model

        self.scaler = StandardScaler()

        self.attach_classifier(classifier)

    def attach_classifier(self, classifier: BaseEstimator):
        """Attach an estimator to the model for classification. Required for `self.test_step()`

        Args:
            classifier: the classifier (can also be an Iterable of classifiers) to use at the end of the SCNN
        """
        self.classifier = classifier

    def set_normalize(self, x: np.ndarray):
        self.scaler.fit(x.squeeze())
        return self.normalize(x)

    def normalize(self, x: np.ndarray):
        orig_shape = x.shape
        return self.scaler.transform(x.squeeze()).reshape(orig_shape)

    def predict_embeddings(self, x: np.ndarray | torch.Tensor):
        if len(x.shape) == 3:
            x = x.reshape(-1, 1, *x.shape[1:])
        elif len(x.shape) == 2:
            x = x.reshape(-1, 1, 1, x.shape[1])
        x = self.normalize(x)
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).to(self.model.device)
        x = x.float()
        with torch.no_grad():
            return self.model(x).cpu().detach().numpy()

    def fit(self, x, y):
        """
        Fit the output classifier on the given data.

        Args:
            x: numpy data that is passed through the CNN before fitting
            y: labels
        """
        self.model.eval()
        embeddings = self.predict_embeddings(x)
        self.classifier.fit(embeddings, y)

    def predict_proba(self, x):
        embeddings = self.predict_embeddings(x)
        return self.classifier.predict_proba(embeddings)

    def predict(self, x):
        embeddings = self.predict_embeddings(x)
        return self.classifier.predict(embeddings)

    @staticmethod
    def load_from_disk(model_path: str, emg_shape: tuple, accelerator: str = "cpu"):
        """
        Load an SCNN model from disk in evaluation mode.
        """
        chkpt = torch.load(model_path)

        model = EmgSCNN(emg_shape)
        model.load_state_dict(chkpt["model_state_dict"])

        mw = EmgSCNNWrapper(model.to(accelerator).eval(), chkpt["classifier"])
        mw.scaler = chkpt["scaler"]

        print(
            f"Loaded SCNN model from {model_path}. Classifier is {mw.classifier.__class__.__name__}"
        )

        return mw

    def save_to_disk(self, model_path: str):
        """
        Save the SCNN model to disk.
        """
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "classifier": self.classifier,
                "scaler": self.scaler,
            },
            model_path,
        )
        print(
            f"Saved SCNN model to {model_path}. Classifier is {self.classifier.__class__.__name__}"
        )


def save_nn(model: EmgCNN | EmgMLP, out_path: str):
    print(f"Saving model to {out_path}")
    torch.save(
        {"model_state_dict": model.state_dict(), "scaler": model.scaler}, out_path
    )


def load_mlp(model_path: str):
    """
    Load a model checkpoint from path and return it, including the StandardScaler
    """
    log.info(f"Loading MLP from {model_path}")
    chkpt = torch.load(model_path)
    s_dict = chkpt["model_state_dict"]
    n_input = s_dict["feature_extractor.1.weight"].shape[1]
    n_classes = s_dict["classifier.weight"].shape[0]
    model = EmgMLP(n_input, n_classes)
    model.load_state_dict(s_dict)
    model.scaler = chkpt["scaler"]
    return model.eval()


def load_conv(model_path: str, num_channels: int, emg_shape: tuple):
    """
    Load a model checkpoint from path and return it, including the StandardScaler
    """
    log.info(f"Loading CNN from {model_path}")
    chkpt = torch.load(model_path)
    s_dict = chkpt["model_state_dict"]
    n_classes = s_dict["classifier.weight"].shape[0]
    model = EmgCNN(num_channels, emg_shape, n_classes)
    model.load_state_dict(s_dict)
    model.scaler = chkpt["scaler"]
    return model.eval()


def train_nn(
    model: EmgCNN | EmgMLP,
    sensor: EmgSensor,
    features: list,
    data_dir: str,
    classes: list,
    train_reps: list,
    test_reps: list,
    finetune: bool = False,
):
    """
    Train/finetune a NN model

    Returns the trained model
    """
    if not isinstance(train_reps, Iterable):
        train_reps = [train_reps]
    if not isinstance(test_reps, Iterable):
        test_reps = [test_reps]

    odh = datasets.get_offline_datahandler(data_dir, classes, train_reps + test_reps)
    train_odh = odh.isolate_data("reps", train_reps)
    val_odh = odh.isolate_data("reps", test_reps)

    # fi = utils.get_filter(sensor.fs, sensor.bandpass_freqs)
    # fi.filter(train_odh)
    # fi.filter(test_odh)

    train_win, train_labels = datasets.prepare_data(train_odh, sensor)
    train_data = FeatureExtractor().extract_features(features, train_win, array=True)
    train_data = (
        model.scaler.fit_transform(train_data)
        if not finetune
        else model.scaler.transform(train_data)
    )
    train_loader = datasets.get_dataloader(
        train_data.astype(np.float32), train_labels, 64, True
    )

    val_loader = None
    if len(test_reps) > 0:
        val_win, val_labels = datasets.prepare_data(val_odh, sensor)
        val_data = FeatureExtractor().extract_features(features, val_win, array=True)
        val_data = model.scaler.transform(val_data)
        val_loader = datasets.get_dataloader(
            val_data.astype(np.float32), val_labels, 256, False
        )

    trainer = L.Trainer(
        max_epochs=15, callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)]
    )
    trainer.fit(model, train_loader, val_loader)

    return model


def main_train_nn(
    model: L.LightningModule,
    sensor: EmgSensor,
    sample_data: bool,
    features: list,
    gestures_list: list,
    gestures_dir: str,
    data_dir: str,
    model_out_path: str,
    num_reps: int,
    rep_time: int,
):
    if sample_data:
        utils.do_sgt(sensor, gestures_list, gestures_dir, data_dir, num_reps, rep_time)

    classes = utils.get_cid_from_gid(gestures_dir, data_dir, gestures_list)
    reps = utils.get_reps(data_dir)
    if len(reps) == 1:
        train_reps = reps
        test_reps = []
    else:
        train_reps = reps[: int(0.8 * len(reps))]
        test_reps = reps[int(0.8 * len(reps)) :]

    model = train_nn(
        model,
        sensor,
        features,
        data_dir,
        classes,
        train_reps,
        test_reps,
        True if num_reps < 3 else False,
    )
    save_nn(model, model_out_path)
    return model


def main_test_nn(
    model: EmgCNN | EmgMLP,
    sensor: EmgSensor,
    sample_data: bool,
    features: list,
    gestures_list: list,
    gestures_dir: str,
    data_dir: str,
):
    if sample_data:
        utils.do_sgt(sensor, gestures_list, gestures_dir, data_dir, 2, 3)

    classes = utils.get_cid_from_gid(gestures_dir, data_dir, gestures_list)
    reps = utils.get_reps(data_dir)
    idle_cid = utils.map_gid_to_cid(gestures_dir, data_dir)[1]

    odh = datasets.get_offline_datahandler(data_dir, classes, reps)
    data, labels = datasets.prepare_data(odh, sensor)
    data = FeatureExtractor().extract_features(features, data, array=True)

    classifier = EMGClassifier()
    classifier.classifier = model.eval()
    # classifier.add_majority_vote(sensor.maj_vote_n)
    # classifier.add_rejection(0.9)

    preds, _ = classifier.run(data)

    om = OfflineMetrics()
    metrics = ["CA", "AER", "INS", "REJ_RATE", "CONF_MAT"]
    results = om.extract_offline_metrics(metrics, labels, preds, idle_cid)

    for key in results:
        if key == "CONF_MAT":
            continue
        print(f"{key}: {results[key]}")

    return results


def train_scnn(
    mw: EmgSCNNWrapper,
    sensor: EmgSensor,
    data_dir: str,
    classes: list,
    train_reps: list,
    val_reps: list,
):
    """
    Train a SCNN model

    Returns the trained model
    """
    if not isinstance(train_reps, Iterable):
        train_reps = [train_reps]
    if not isinstance(val_reps, Iterable):
        val_reps = [val_reps]

    odh = datasets.get_offline_datahandler(data_dir, classes, train_reps + val_reps)
    train_odh = odh.isolate_data("reps", train_reps)
    val_odh = odh.isolate_data("reps", val_reps)

    # fi = utils.get_filter(sensor.fs, sensor.bandpass_freqs, sensor.notch_freq)
    # fi.filter(train_odh)
    # fi.filter(val_odh)

    # Generate triplets and train

    train_windows, train_labels = datasets.prepare_data(train_odh, sensor)
    train_data = FeatureExtractor().getMAVfeat(train_windows)
    fit_data = np.copy(train_data)
    train_data = mw.scaler.fit_transform(train_data)
    train_data = np.reshape(train_data, (-1, 1, *sensor.emg_shape))

    val_windows, val_labels = datasets.prepare_data(val_odh, sensor)
    val_data = FeatureExtractor().getMAVfeat(val_windows)
    val_data = mw.scaler.transform(val_data)
    val_data = np.reshape(val_data, (-1, 1, *sensor.emg_shape))

    train_loader = datasets.get_triplet_dataloader(
        train_data.astype(np.float32),
        train_labels,
        32,
        True,
        len(train_data) // (3 * len(classes)),
    )
    val_loader = datasets.get_triplet_dataloader(
        val_data.astype(np.float32),
        val_labels,
        256,
        False,
        len(val_data) // (3 * len(classes)),
    )

    trainer = L.Trainer(max_epochs=15)
    trainer.fit(mw.model, train_loader, val_loader)

    mw.model.eval()
    mw.fit(
        fit_data, train_labels
    )  # Fit output classifier and bypass double-scaling of data
    return mw


def main_train_scnn(
    sensor: EmgSensor,
    data_dir: str,
    sample_data: bool,
    gestures_list: list,
    gestures_dir: str,
    classifier: BaseEstimator,
):
    """
    Train SCNN. The SCNN's classifier is `fit` on the train data.
    """
    if sample_data:
        utils.do_sgt(sensor, gestures_list, gestures_dir, data_dir, 5, 3)

    classes = utils.get_cid_from_gid(gestures_dir, data_dir, gestures_list)
    reps = utils.get_reps(data_dir)
    if len(reps) == 1:
        train_reps = reps
        val_reps = []
    else:
        train_reps = reps[: int(0.8 * len(reps))]
        val_reps = reps[int(0.8 * len(reps)) :]

    model = EmgSCNN(sensor.emg_shape)
    mw = EmgSCNNWrapper(model, classifier)
    mw = train_scnn(mw, sensor, data_dir, classes, train_reps, val_reps)
    return mw


def main_finetune_scnn(
    mw: EmgSCNNWrapper,
    sensor: EmgSensor,
    data_dir: str,
    sample_data: bool,
    gestures_list: list,
    gestures_dir: str,
):
    if sample_data:
        utils.do_sgt(sensor, gestures_list, gestures_dir, data_dir, 1, 2)

    classes = utils.get_cid_from_gid(gestures_dir, data_dir, gestures_list)
    reps = utils.get_reps(data_dir)

    fe = FeatureExtractor()

    odh = datasets.get_offline_datahandler(data_dir, classes, reps)
    data_windows, labels = datasets.prepare_data(odh, sensor)
    data = fe.getMAVfeat(data_windows)
    data = np.reshape(data, (-1, 1, *sensor.emg_shape))

    # Fit classifier
    mw.fit(data, labels)
    return mw


def main_test_scnn(
    mw: EmgSCNNWrapper,
    sensor: EmgSensor,
    data_dir: str,
    sample_data: bool,
    gestures_list: list,
    gestures_dir: str,
):
    if sample_data:
        utils.do_sgt(sensor, gestures_list, gestures_dir, data_dir, 2, 3)

    classes = utils.get_cid_from_gid(gestures_dir, data_dir, gestures_list)
    idle_id = utils.map_gid_to_cid(gestures_dir, data_dir)[1]
    reps = utils.get_reps(data_dir)

    odh = datasets.get_offline_datahandler(data_dir, classes, reps)
    # fi = utils.get_filter(sensor.fs, sensor.bandpass_freqs, sensor.notch_freq)
    # fi.filter(odh)

    data_windows, test_labels = datasets.prepare_data(odh, sensor)
    # data = FeatureExtractor().getMEANfeat(data_windows)
    data = FeatureExtractor().getMAVfeat(data_windows)

    mw.model.eval()

    preds = mw.predict(data)
    preds_maj = majority_vote(preds, sensor.maj_vote_n)

    acc = accuracy_score(test_labels, preds)
    acc_maj = accuracy_score(test_labels, preds_maj)

    print(f"Raw accuracy: {acc*100:.2f}")
    print(f"Majority vote accuracy: {acc_maj*100:.2f}")

    classifier = EMGClassifier()
    classifier.classifier = mw
    classifier.add_majority_vote(sensor.maj_vote_n)
    # classifier.add_rejection(0.8)

    preds, _ = classifier.run(data)
    # for i in range(len(set(test_labels))):
    #     print(set(preds[test_labels == i]))
    # print(set(preds))

    # https://libemg.github.io/libemg/documentation/evaluation/evaluation.html
    om = OfflineMetrics()
    metrics = ["CA", "AER", "REJ_RATE", "CONF_MAT"]
    results = om.extract_offline_metrics(
        metrics, test_labels, preds, null_label=idle_id
    )

    for key in results:
        if key == "CONF_MAT":
            continue
        print(f"{key}: {results[key]}")
    return results
