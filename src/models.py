import numpy as np
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F

from sklearn.metrics import accuracy_score


class EmgCNN(L.LightningModule):
    def __init__(self, input_shape, num_classes):
        """
        Parameters:
            - input_shape: shape of input data
            - num_classes: number of classes
        """
        super().__init__()
        self.save_hyperparameters()
        self.input_shape = input_shape

        hidden_layer_sizes = [32, 32, 32, 256]

        # Conv layers
        self.conv1 = nn.Conv2d(1, hidden_layer_sizes[0], 3, padding=1)
        self.relu1 = nn.ReLU()
        self.bn1 = nn.BatchNorm2d(hidden_layer_sizes[0])

        self.conv2 = nn.Conv2d(
            hidden_layer_sizes[0], hidden_layer_sizes[1], 3, padding=1
        )
        self.relu2 = nn.ReLU()
        self.bn2 = nn.BatchNorm2d(hidden_layer_sizes[1])

        self.conv3 = nn.Conv2d(
            hidden_layer_sizes[1], hidden_layer_sizes[2], 5, padding=2
        )
        self.relu3 = nn.ReLU()
        self.bn3 = nn.BatchNorm2d(hidden_layer_sizes[2])

        # Fully connected layers
        self.flat = nn.Flatten()

        self.fc4 = nn.Linear(
            hidden_layer_sizes[2] * np.prod(self.input_shape),
            hidden_layer_sizes[3],
        )
        self.dropout4 = nn.Dropout(0.5)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(hidden_layer_sizes[3])

        self.fc5 = nn.Linear(hidden_layer_sizes[3], num_classes)

    def forward(self, x):
        out = torch.reshape(x, (-1, 1, *self.input_shape))
        out = self.bn1(self.relu1(self.conv1(out)))
        out = self.bn2(self.relu2(self.conv2(out)))
        out = self.bn3(self.relu3(self.conv3(out)))
        out = self.flat(out)
        out = self.bn4(self.relu4(self.dropout4(self.fc4(out))))
        logits = self.fc5(out)
        return logits

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = F.cross_entropy(y, y_true)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = F.cross_entropy(y, y_true)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y: torch.Tensor = self(x)
        loss: float = F.cross_entropy(y, y_true)

        y = np.argmax(y.cpu().detach().numpy(), axis=1)
        y_true = y_true.cpu().detach().numpy()

        acc = accuracy_score(y_true, y, normalize=True)

        self.log("test_acc", acc)
        self.log("test_loss", loss)
        return acc, loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer


class TuningEmgCNN(L.LightningModule):
    def __init__(self, model: EmgCNN, num_classes: int):
        """
        Parameters:
            - num_classes: number of classes
        """
        super().__init__()

        # Take the feature extractor from the model, pop the last classification layer
        num_filters = model.fc5.in_features
        layers = list(model.children())[:-1]

        print(list(model.children()))

        self.feature_extractor = nn.Sequential(*layers)
        self.feature_extractor.eval()

        # Add a new classification layer
        self.classifier = nn.Linear(num_filters, num_classes)

    def forward(self, x):
        representations: torch.Tensor
        with torch.no_grad():
            representations = self.feature_extractor(x)
        x = self.classifier(representations)
        return x

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = F.cross_entropy(y, y_true)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y = self(x)
        loss = F.cross_entropy(y, y_true)
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        # training_step defines the train loop. It is independent of forward
        x, y_true = batch
        y: torch.Tensor = self(x)
        loss: float = F.cross_entropy(y, y_true)

        y = np.argmax(y.cpu().detach().numpy(), axis=1)
        y_true = y_true.cpu().detach().numpy()

        acc = accuracy_score(y_true, y, normalize=True)

        self.log("test_acc", acc)
        self.log("test_loss", loss)
        return acc, loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-3)
        return optimizer
