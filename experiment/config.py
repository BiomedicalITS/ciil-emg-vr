from enum import IntEnum
import numpy as np
import torch

import libemg

from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths
from nfc_emg import models


class ExperimentStage(IntEnum):
    FAMILIARIZATION = 0
    SG_TRAIN = 1
    SG_TEST = 2
    GAME = 3
    SG_POST_TEST = 4
    VISUALIZE_CLASSIFIER = 5


class Config:
    def __init__(
        self,
        subject_id: int,
        sensor_type: EmgSensorType,
        adaptation: bool,
        stage: ExperimentStage,
    ):
        self.stage = stage
        self.subject_id = subject_id
        self.sensor = EmgSensor(
            sensor_type, window_size_ms=150, window_inc_ms=20, majority_vote_ms=0
        )

        self.adaptation = adaptation

        self.model_type = "CNN"  # Can be "CNN" or "MLP"
        self.negative_method = "mixed"  # Can be "mixed" or "none"
        self.relabel_method = "LabelSpreading"

        # 24 is Ring Flexion in LibEMG but image is tripod pinch
        self.gesture_ids = [1, 2, 3, 4, 5, 8, 26, 30]
        self.features = "TDPSD"  # Can be list of features OR feature group

        self.get_feature_parameters()
        self.get_path_parameters()
        self.get_datacollection_parameters()
        self.get_classifier_parameters()
        self.get_game_parameters()

    def get_path_parameters(self):
        self.paths = NfcPaths(f"data/{self.subject_id}/", self.sensor.get_name())
        self.paths.gestures = "data/gestures/"

    def get_feature_parameters(self):
        fe = libemg.feature_extractor.FeatureExtractor()
        if isinstance(self.features, str):
            self.features = fe.get_feature_groups()[self.features]

        # fake_window = np.random.randn(1, np.prod(self.sensor.emg_shape), self.sensor.window_size)
        # returned_features = fe.extract_features(self.features, fake_window)

        # self.input_shape = np.squeeze(self.sensor.emg_shape)
        # self.num_features = len(returned_features)

    def get_datacollection_parameters(self):
        self.reps = 5
        self.rep_time = 3
        self.rest_time = 1

    def get_classifier_parameters(self):
        self.oc_output_format = "probabilities"

        self.input_shape = self.sensor.emg_shape
        self.num_channels = len(self.features)

        self.accelerator = (
            "cuda"
            if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available() else "cpu"
        )

        if self.stage == ExperimentStage.SG_TRAIN:
            # New model
            if self.model_type == "CNN":
                self.model = models.EmgCNN(
                    len(self.features), self.sensor.emg_shape, len(self.gesture_ids)
                )
            elif self.model_type == "MLP":
                self.model = models.EmgMLP(
                    len(self.features) * np.prod(self.sensor.emg_shape),
                    len(self.gesture_ids),
                )
            else:
                raise ValueError("Invalid model type.")
            return
        elif self.stage == ExperimentStage.SG_POST_TEST:
            self.paths.set_model("model_post")

        # Load if needed
        if self.model_type == "CNN":
            self.model = models.load_conv(
                self.paths.get_model(), self.num_channels, self.input_shape
            )
        elif self.model_type == "MLP":
            self.model = models.load_mlp(self.paths.get_model())
        else:
            raise ValueError("Invalid model type.")

        self.model.to(self.accelerator)

    def get_game_parameters(self):
        self.game_time = 600
