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
        self.fine_tuning = False

        hidden_layer_sizes = [32, 32, 32, 128, 64]

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
            hidden_layer_sizes[2] * np.prod(input_shape),
            hidden_layer_sizes[3],
        )
        self.do4 = nn.Dropout(0.5)
        self.relu4 = nn.ReLU()
        self.bn4 = nn.BatchNorm1d(hidden_layer_sizes[3])

        self.fc5 = nn.Linear(
            hidden_layer_sizes[3],
            hidden_layer_sizes[4],
        )
        self.do5 = nn.Dropout(0.5)
        self.relu5 = nn.ReLU()
        self.bn5 = nn.BatchNorm1d(hidden_layer_sizes[4])

        self.classifier = nn.Linear(hidden_layer_sizes[4], num_classes)

    def _feature_extractor(self, x):
        out = self.bn1(self.relu1(self.conv1(x)))
        out = self.bn2(self.relu2(self.conv2(out)))
        out = self.bn3(self.relu3(self.conv3(out)))
        out = self.flat(out)
        out = self.bn4(self.do4(self.relu4(self.fc4(out))))
        return out

    def forward(self, x):
        if self.fine_tuning:
            with torch.no_grad():
                out = self._feature_extractor(x)
        else:
            out = self._feature_extractor(x)

        out = self.bn5(self.do5(self.relu5(self.fc5(out))))
        logits = self.classifier(out)
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

    def set_finetune(self, fine_tuning: bool, num_classes: int):
        self.fine_tuning = fine_tuning
        if num_classes != self.classifier.out_features:
            print(
                f"Setting to {num_classes} classes from {self.classifier.out_features}"
            )
            self.classifier = nn.Linear(self.classifier.in_features, num_classes)
