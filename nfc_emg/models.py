import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from typing import Iterable

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator

from libemg.offline_metrics import OfflineMetrics
from libemg.emg_classifier import EMGClassifier
from libemg.feature_extractor import FeatureExtractor

from emager_py.majority_vote import majority_vote

from nfc_emg import datasets, utils
from nfc_emg.sensors import EmgSensor, EmgSensorType


class EmgCNN(L.LightningModule):
    def __init__(self, input_shape, num_classes):
        """
        Parameters:
            - input_shape: shape of input data
            - num_classes: number of classes
        """
        super().__init__()
        self.save_hyperparameters()
        self.fine_tuning = False

        hl_sizes = [32, 32, 32, 128, 64]

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, hl_sizes[0], 5, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(hl_sizes[0]),
            nn.Conv2d(hl_sizes[0], hl_sizes[1], 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(hl_sizes[1]),
            nn.Conv2d(hl_sizes[1], hl_sizes[2], 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(hl_sizes[2]),
            nn.Flatten(),
            nn.Linear(hl_sizes[2] * np.prod(input_shape), hl_sizes[3]),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hl_sizes[3]),
        )

        self.fine_feature_extractor = nn.Sequential(
            nn.Linear(hl_sizes[3], hl_sizes[4]),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hl_sizes[4]),
        )

        self.classifier = nn.Linear(hl_sizes[4], num_classes)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.fine_feature_extractor(out)
        logits = self.classifier(out)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = F.cross_entropy(y, y_true)

        acc = accuracy_score(
            y_true.cpu().detach().numpy(),
            np.argmax(y.cpu().detach().numpy(), axis=1),
            normalize=True,
        )
        self.log("train_loss", loss)
        self.log("train_acc", acc)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = F.cross_entropy(y, y_true)
        acc = accuracy_score(
            y_true.cpu().detach().numpy(),
            np.argmax(y.cpu().detach().numpy(), axis=1),
            normalize=True,
        )
        self.log("val_loss", loss)
        self.log("val_acc", acc)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y: torch.Tensor = self(x)
        loss: float = F.cross_entropy(y, y_true)

        y = np.argmax(y.cpu().detach().numpy(), axis=1)
        y_true = y_true.cpu().detach().numpy()

        acc = accuracy_score(y_true, y, normalize=True)
        try:
            self.log("test_loss", loss)
            self.log("test_acc", acc)
        finally:
            return {"loss": loss, "y_true": list(y_true), "y_pred": list(y)}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer

    def convert_input(self, x):
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).to(self.device)
        if len(x.shape) == 3:
            x = x.reshape(-1, 1, *x.shape[1:])
        elif len(x.shape) == 2:
            x = x.reshape(-1, 1, 1, x.shape[1])
        return x

    def predict(self, x):
        x = self.convert_input(x)
        with torch.no_grad():
            return np.argmax(self(x).cpu().detach().numpy(), axis=1)

    def fit(self, x, y):
        x = self.convert_input(x)
        if not isinstance(y, torch.Tensor):
            y = torch.from_numpy(y).to(self.device)
        self.training_step(x, y)

    def set_finetune(self, fine_tuning: bool, num_classes: int):
        self.fine_tuning = fine_tuning
        for param in self.feature_extractor.parameters():
            param.requires_grad_(fine_tuning)
            print(param.requires_grad)
        if num_classes != self.classifier.out_features:
            print(
                f"Setting to {num_classes} classes from {self.classifier.out_features}"
            )
            self.classifier = nn.Linear(self.classifier.in_features, num_classes)


class EmgSCNN(L.LightningModule):
    def __init__(self, input_shape):
        """
        Parameters:
            - input_shape: EMG input shape (H, W)
            - classifier: a classifier to attach to the model. Can also be attached later with `attach_classifier()`
        """
        super().__init__()
        self.save_hyperparameters()

        hl_sizes = [32, 32, 32, 128, 64]

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(1, hl_sizes[0], 5, padding=2),
            nn.LeakyReLU(),
            nn.BatchNorm2d(hl_sizes[0]),
            nn.Conv2d(hl_sizes[0], hl_sizes[1], 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(hl_sizes[1]),
            nn.Conv2d(hl_sizes[1], hl_sizes[2], 3, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm2d(hl_sizes[2]),
            nn.Flatten(),
            nn.Linear(hl_sizes[2] * np.prod(input_shape), hl_sizes[3]),
            nn.LeakyReLU(),
            nn.BatchNorm1d(hl_sizes[3]),
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
        self.mean = 0.0
        self.std = 1.0
        self.attach_classifier(classifier)

    def attach_classifier(self, classifier: BaseEstimator):
        """Attach an estimator to the model for classification. Required for `self.test_step()`

        Args:
            classifier: the classifier (can also be an Iterable of classifiers) to use at the end of the SCNN
        """
        self.classifier = classifier

    def set_normalize(self, x):
        self.mean = np.mean(x)
        self.std = np.std(x)
        return self.normalize(x)

    def normalize(self, x):
        return (x - self.mean) / self.std

    def predict_embeddings(self, x):
        x = self.normalize(x)
        if not isinstance(x, torch.Tensor):
            x = torch.from_numpy(x).to(self.model.device)
        if len(x.shape) == 3:
            x = x.reshape(-1, 1, *x.shape[1:])
        elif len(x.shape) == 2:
            x = x.reshape(-1, 1, 1, x.shape[1])
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
        model.to(accelerator)
        model.eval()

        mw = EmgSCNNWrapper(model, chkpt["classifier"])
        mw.mean = chkpt["mean"]
        mw.std = chkpt["std"]

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
                "mean": self.mean,
                "std": self.std,
            },
            model_path,
        )
        print(
            f"Saved SCNN model to {model_path}. Classifier is {self.classifier.__class__.__name__}"
        )


def get_model(model_path: str, emg_shape: tuple, num_classes: int, finetune: bool):
    """
    Load a model checkpoint from path and return it
    """
    print(f"Loading model from {model_path}")
    chkpt = torch.load(model_path)
    n_classes = chkpt["classifier.weight"].shape[0]
    model = EmgCNN(emg_shape, n_classes)
    model.load_state_dict(chkpt)
    model.set_finetune(finetune, num_classes)
    return model.eval()


def save_cnn(model: EmgCNN, out_path: str):
    print(f"Saving CNN model to {out_path}.")
    torch.save("model_state_dict", model.state_dict(), out_path)


def train_cnn(
    model: EmgCNN,
    sensor: EmgSensor,
    data_dir: str,
    classes: list,
    train_reps: list,
    test_reps: list,
):
    """Train a CNN model

    Args:
        data_dir (str): directory of pre-recorded data
        classes: class ids (CIDs)
        train_reps (list): which repetitions to train on
        test_reps (list): which repetitions to test on (can be empty)
        moving_avg_n (int): _description_

    Returns the trained model
    """
    if not isinstance(train_reps, Iterable):
        train_reps = [train_reps]
    if not isinstance(test_reps, Iterable):
        test_reps = [test_reps]

    odh = datasets.get_offline_datahandler(data_dir, classes, train_reps + test_reps)
    train_odh = odh.isolate_data("reps", train_reps)
    test_odh = odh.isolate_data("reps", test_reps)

    data, labels = datasets.prepare_data(train_odh, sensor)
    train_loader = datasets.get_dataloader(data, labels, 64, True)

    model.train()
    trainer = L.Trainer(
        max_epochs=10,
        callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)],
    )
    trainer.fit(model, train_loader)

    if len(test_reps) > 0:
        data, labels = datasets.prepare_data(test_odh, sensor)
        test_loader = datasets.get_dataloader(data, labels, 128, False)
        trainer.test(model, test_loader)

    return model


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

    # Generate triplets and train
    fe = FeatureExtractor()

    train_windows, train_labels = datasets.prepare_data(train_odh, sensor)
    train_data = fe.extract_features(["MAV"], train_windows, array=True)
    train_data = np.reshape(train_data, (-1, *sensor.emg_shape)).astype(np.float32)
    train_data = mw.set_normalize(train_data)

    val_windows, val_labels = datasets.prepare_data(val_odh, sensor)
    val_data = fe.extract_features(["MAV"], val_windows, array=True)
    val_data = np.reshape(val_data, (-1, *sensor.emg_shape)).astype(np.float32)
    val_data = mw.normalize(val_data)

    train_loader = datasets.get_triplet_dataloader(
        train_data, train_labels, 64, True, len(train_data) // (3 * len(classes))
    )
    val_loader = datasets.get_triplet_dataloader(
        val_data, val_labels, 256, False, len(val_data) // (3 * len(classes))
    )

    trainer = L.Trainer(
        max_epochs=10,
        callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)],
        deterministic=True,
    )
    trainer.fit(mw.model, train_loader, val_loader)
    mw.model.eval()
    return mw


def test_model(
    model: EmgCNN,
    sensor: EmgSensor,
    data_dir: str,
    classes: list,
    test_reps: list,
):
    """Test model. Returns (y_pred, y_true)."""
    if not isinstance(test_reps, list):
        test_reps = [test_reps]

    odh = datasets.get_offline_datahandler(data_dir, classes, test_reps)
    data, labels = datasets.prepare_data(odh, sensor, 1, 1)
    test_loader = datasets.get_dataloader(data, labels, 128, False)

    model.eval()
    y_pred, y_true = [], []
    with torch.no_grad():
        for i, batch in enumerate(test_loader):
            batch = [b.to(model.device) for b in batch]
            ret = model.test_step(batch, i)
            y_pred.extend(ret["y_pred"])
            y_true.extend(ret["y_true"])
    return np.array(y_pred), np.array(y_true)


def main_train_scnn(
    sensor: EmgSensor,
    data_dir: str,
    sample_data: bool,
    gestures_list: list,
    gestures_dir: str,
    classifier: BaseEstimator,
):
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
    data = fe.extract_features(["MAV"], data_windows, array=True)
    data = data.reshape(-1, *sensor.emg_shape).astype(np.float32)

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

    fe = FeatureExtractor()

    odh = datasets.get_offline_datahandler(data_dir, classes, reps)
    data_windows, test_labels = datasets.prepare_data(odh, sensor)
    data = fe.extract_features(["MAV"], data_windows, array=True)
    data = data.reshape(-1, *sensor.emg_shape).astype(np.float32)

    mw.model.eval()

    preds = mw.predict(data)
    preds_maj = majority_vote(preds, sensor.maj_vote_n)

    acc = accuracy_score(test_labels, preds)
    acc_maj = accuracy_score(test_labels, preds_maj)
    print("Raw accuracy:", acc * 100)
    print("Majority vote accuracy:", acc_maj * 100)

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
    metrics = ["CA", "AER", "REJ_RATE", "CONF_MAT", "RECALL", "PREC"]

    results = om.extract_offline_metrics(
        metrics, test_labels, preds, null_label=idle_id
    )

    for key in results:
        if key == "CONF_MAT":
            continue
        print(f"{key}: {results[key]}")
    return results


def main_cnn(
    sensor: EmgSensor,
    sample_data: bool,
    gestures_list: list,
    gestures_dir: str,
    data_dir: str,
    finetune: bool,
    model_out_path: str,
):
    if sample_data:
        utils.do_sgt(sensor, gestures_list, gestures_dir, data_dir, 5, 5)

    classes = utils.get_cid_from_gid(gestures_dir, data_dir, gestures_list)
    reps = utils.get_reps(data_dir)
    if len(reps) == 1:
        train_reps = reps
        test_reps = []
    else:
        train_reps = reps[: int(0.8 * len(reps))]
        test_reps = reps[int(0.8 * len(reps)) :]

    model = (
        EmgCNN(sensor.emg_shape, len(gestures_list))
        if not finetune
        else get_model(model_out_path, sensor.emg_shape, len(gestures_list), True)
    )
    model = train_cnn(model, sensor, data_dir, classes, train_reps, test_reps)
    torch.save(model.state_dict(), model_out_path)
    return model


def main_test_cnn(
    model: EmgCNN,
    sensor: EmgSensor,
    data_dir: str,
    sample_data: bool,
    gestures_list: list,
    gestures_dir: str,
):
    if sample_data:
        utils.do_sgt(sensor, gestures_list, gestures_dir, data_dir, 2, 3)

    classes = utils.get_cid_from_gid(gestures_dir, data_dir, gestures_list)
    reps = utils.get_reps(data_dir)
    idle_cid = utils.map_gid_to_cid(gestures_dir, data_dir)[1]

    preds, labels = test_model(model, sensor, data_dir, classes, reps)
    preds_maj = majority_vote(preds, sensor.maj_vote_n)

    om = OfflineMetrics()
    metrics = ["CA", "AER", "INS", "REJ_RATE", "CONF_MAT", "RECALL", "PREC", "F1"]
    results = om.extract_offline_metrics(metrics, labels, preds, idle_cid)
    results_maj = om.extract_offline_metrics(metrics, labels, preds_maj, idle_cid)

    print(f"Precision RAW: {results['PREC']}")
    print(f"Precision MAJ: {results_maj['PREC']}")

    return results_maj
