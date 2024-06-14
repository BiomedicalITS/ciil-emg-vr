from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from sklearn.metrics import accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.utils import shuffle
from sklearn.base import BaseEstimator

from nfc_emg import datasets
from nfc_emg.sensors import EmgSensor


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

        hidden_layer_sizes = [64, 32, 32, 128, 64]

        # Conv layers
        self.conv1 = nn.Conv2d(1, hidden_layer_sizes[0], 5, padding=2)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(hidden_layer_sizes[0])

        self.conv2 = nn.Conv2d(
            hidden_layer_sizes[0], hidden_layer_sizes[1], 3, padding=1
        )
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(hidden_layer_sizes[1])

        self.conv3 = nn.Conv2d(
            hidden_layer_sizes[1], hidden_layer_sizes[2], 3, padding=1
        )
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(hidden_layer_sizes[2])

        # Fully connected layers
        self.flat = nn.Flatten()

        self.fc4 = nn.Linear(
            hidden_layer_sizes[2] * np.prod(input_shape),
            hidden_layer_sizes[3],
        )
        self.do4 = nn.Dropout(0.5)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(hidden_layer_sizes[3])

        self.feature_extractor = nn.Sequential(
            self.conv1,
            self.relu1,
            self.bn1,
            self.conv2,
            self.relu2,
            self.bn2,
            self.conv3,
            self.relu3,
            self.bn3,
            self.flat,
            self.fc4,
            self.do4,
            self.relu4,
            self.bn4,
        )

        self.fc5 = nn.Linear(
            hidden_layer_sizes[3],
            hidden_layer_sizes[4],
        )
        self.do5 = nn.Dropout(0.5)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm1d(hidden_layer_sizes[4])

        self.classifier = nn.Linear(hidden_layer_sizes[4], num_classes)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = self.bn5(self.do5(self.relu5(self.fc5(out))))
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

    def set_finetune(self, fine_tuning: bool, num_classes: int):
        self.fine_tuning = fine_tuning
        self.feature_extractor.requires_grad = fine_tuning
        if num_classes != self.classifier.out_features:
            print(
                f"Setting to {num_classes} classes from {self.classifier.out_features}"
            )
            self.classifier = nn.Linear(self.classifier.in_features, num_classes)


class EmgSCNN(L.LightningModule):
    def __init__(self, input_shape):
        """
        Create a reference Fully Convolutional Siamese Emage model.

        Parameters:
            - input_shape: EMG input shape (H, W)
            - classifier: a classifier to attach to the model. Can also be attached later with `attach_classifier()`
        """
        super().__init__()
        self.save_hyperparameters()

        output_sizes = [32, 32, 32, 128, 64]

        self.conv1 = nn.Conv2d(1, output_sizes[0], 5, padding=2)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(output_sizes[0])

        self.conv2 = nn.Conv2d(output_sizes[0], output_sizes[1], 3, padding=1)
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(output_sizes[1])

        self.conv3 = nn.Conv2d(output_sizes[1], output_sizes[2], 3, padding=1)
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(output_sizes[2])

        self.flat = nn.Flatten()

        self.fc4 = nn.Linear(
            output_sizes[2] * np.prod(input_shape),
            output_sizes[3],
        )
        self.do4 = nn.Dropout(0.5)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(output_sizes[3])

        self.fc5 = nn.Linear(
            output_sizes[3],
            output_sizes[4],
        )

    def forward(self, x):
        out = self.bn1(self.relu1(self.conv1(x)))
        out = self.bn2(self.relu2(self.conv2(out)))
        out = self.bn3(self.relu3(self.conv3(out)))
        out = self.flat(out)
        out = self.bn4(self.do4(self.relu4(self.fc4(out))))
        features = self.fc5(out)
        return features

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
    def __init__(self):
        """
        Create a cosine similarity classifier.
        """
        super().__init__()

        self.features = None
        self.n_samples = 0

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

        tmp_features = self.features.copy() * self.n_samples
        for i in range(len(y)):
            tmp_features[y[i]] += X[i]
        self.n_samples += len(y)
        self.features = tmp_features / self.n_samples

    def predict(self, X):
        return self.__cosine_similarity(X, True)

    def predict_proba(self, X):
        return self.__cosine_similarity(X, False)


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

    def attach_classifier(self, classifier):
        """Attach an estimator to the model for classification. Required for `self.test_step()`

        Args:
            classifier: the classifier (can also be an Iterable of classifiers) to use at the end of the SCNN
        """
        if classifier is None:
            return
        elif not isinstance(classifier, Iterable):
            classifier = [classifier]
        self.classifier = make_pipeline(*classifier)

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
        with torch.no_grad():
            return self.model(x).cpu().detach().numpy()

    def fit(self, x, y):
        """
        Fit the output classifier on the given data.

        Args:
            x: numpy data that is passed through the CNN before fitting
            y: labels
        """
        embeddings = self.predict_embeddings(x)
        self.classifier.fit(embeddings, y)

    def predict_proba(self, x):
        embeddings = self.predict_embeddings(x)
        return self.classifier.predict_proba(embeddings)

    def predict(self, x):
        embeddings = self.predict_embeddings(x)
        return self.classifier.predict(embeddings)


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


def save_scnn(mw: EmgSCNNWrapper, out_path: str):
    print(
        f"Saving SCNN model to {out_path}. Classifier is {mw.classifier.steps[-1][1].__class__.__name__}"
    )
    torch.save(
        {
            "model_state_dict": mw.model.state_dict(),
            "classifier": mw.classifier,
            "mean": mw.mean,
            "std": mw.std,
        },
        out_path,
    )


def save_cnn(model: EmgCNN, out_path: str):
    print(f"Saving CNN model to {out_path}.")
    torch.save("model_state_dict", model.state_dict(), out_path)


def get_scnn(model_path: str, emg_shape: tuple, accelerator: str = "cpu"):
    """
    Load an SCNN model.
    """
    print(f"Loading SCNN model from {model_path}")
    chkpt = torch.load(model_path)

    model = EmgSCNN(emg_shape).to(accelerator)
    model.load_state_dict(chkpt["model_state_dict"])

    mw = EmgSCNNWrapper(model, chkpt["classifier"])
    mw.mean = chkpt["mean"]
    mw.std = chkpt["std"]
    return mw


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

    data, labels = datasets.prepare_data(train_odh, sensor, 1, 1)
    train_loader = datasets.get_dataloader(data, labels, 64, True)

    model.train()
    trainer = L.Trainer(
        max_epochs=10,
        callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)],
    )
    trainer.fit(model, train_loader)

    if len(test_reps) > 0:
        data, labels = datasets.prepare_data(test_odh, sensor, 1, 1)
        test_loader = datasets.get_dataloader(data, labels, 128, False)
        trainer.test(model, test_loader)

    return model


def train_scnn(
    mw: EmgSCNNWrapper,
    sensor: EmgSensor,
    data_dir: str,
    classes: list,
    train_reps: list,
    test_reps: list,
):
    """Train a SCNN model

    Args:
        data_dir (str): directory of pre-recorded data
        classes: class ids (CIDs)
        train_reps (list): which repetitions to train on
        test_reps (list): which repetitions to test on (can be empty)
        n_triplets (int): number of triplets to generate

    Returns the trained model
    """
    if not isinstance(train_reps, list):
        train_reps = [train_reps]
    if not isinstance(test_reps, list):
        test_reps = [test_reps]

    odh = datasets.get_offline_datahandler(data_dir, classes, train_reps + test_reps)

    train_odh = odh.isolate_data("reps", train_reps)
    test_odh = odh.isolate_data("reps", test_reps)

    # Generate triplets and train
    num_triplets = 0
    for f in train_odh.data:
        num_triplets += len(f)

    train_data, train_labels = datasets.prepare_data(train_odh, sensor, 1, 1)
    train_data = mw.set_normalize(train_data)
    train_loader = datasets.get_triplet_dataloader(
        train_data, train_labels, 64, True, num_triplets // (3 * len(classes))
    )

    mw.model.train()
    trainer = L.Trainer(
        max_epochs=10,
        callbacks=[EarlyStopping(monitor="train_loss", min_delta=0.0005)],
        deterministic=True,
    )
    trainer.fit(mw.model, train_loader)
    mw.model.eval()

    if len(test_reps) > 0:
        test_data, test_labels = datasets.prepare_data(test_odh, sensor, 1, 1)
        test_data, test_labels = shuffle(test_data, test_labels)
        mw.fit(test_data, test_labels)
        preds = mw.predict(test_data)
        acc = accuracy_score(test_labels, preds)
        print(f"Test accuracy: {acc:.2f}")

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
