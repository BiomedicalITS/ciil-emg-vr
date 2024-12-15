import os
from shutil import copytree
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
        features: str | list,
        stage: ExperimentStage,
        adaptation: bool = True,
    ):
        POWERLINE_NOTCH_FREQ = 60  # 50 for EU, 60 for NA
        self.model_type = "CNN"  # Can be "CNN" or "MLP"
        self.negative_method = "mixed"  # Can be "mixed" or "none"
        self.relabel_method = "LabelSpreading"
        self.gesture_ids = [1, 2, 3, 4, 5, 8, 26, 30]

        self.stage = stage
        self.subject_id = subject_id
        self.sensor = EmgSensor(
            sensor_type,
            notch_freq=POWERLINE_NOTCH_FREQ,
            window_size_ms=200,
            window_inc_ms=50,
            majority_vote_ms=0,
        )

        self.adaptation = adaptation
        self.features = features  # Can be list of features OR feature group

        if self.relabel_method == "LabelSpreading":
            os.environ["OMP_NUM_THREADS"] = "1"

        if not torch.cuda.is_available():
            print("========================================")
            print("CRITICAL WARNING: CUDA is not available.")
            input("Press any key to continue....")
            print("========================================")

        self.get_path_parameters()
        self.get_feature_parameters()
        self.get_datacollection_parameters()
        self.get_classifier_parameters()
        self.get_game_parameters()

    def get_path_parameters(self):
        self.paths = NfcPaths(
            f"data/{self.subject_id}/{self.sensor.get_name()}", "no_adap"
        )
        self.paths.gestures = "data/gestures/"

        if self.stage >= ExperimentStage.GAME and self.adaptation:
            src = self.paths.get_experiment_dir()
            self.paths.trial = "adap"
            if "adap" not in os.listdir(self.paths.base):
                copytree(src, self.paths.get_experiment_dir())

    def get_feature_parameters(self):
        if isinstance(self.features, str):
            fe = libemg.feature_extractor.FeatureExtractor()
            self.features = fe.get_feature_groups()[self.features]

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

        if (
            self.stage == ExperimentStage.SG_TRAIN
            or self.stage == ExperimentStage.FAMILIARIZATION
        ):
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
