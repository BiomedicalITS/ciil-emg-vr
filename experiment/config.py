import os
from enum import IntEnum
import numpy as np
import torch
import shutil
import logging as log

import libemg

from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths
from nfc_emg import models


class ExperimentStage(IntEnum):
    FAMILIARIZATION = 0
    SG_TRAIN = 1
    SG_PRE_TEST = 2
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
        powerline_notch_freq=60,
        model_type="CNN",
        negative_method="mixed",
        relabel_method="none",
        gesture_ids=(1, 2, 3, 4, 5, 8, 26, 30),
        finetune=False,
    ):
        """Create the config experiment.

        Args:
            subject_id (int): Subject ID
            sensor_type (EmgSensorType): Sensor to use
            features (str | list): Features to use, can be a list or a feature group string
            stage (ExperimentStage): Stage of the experiment
            adaptation (bool, optional): Enable adaptation. Defaults to True.
            powerline_notch_freq (int, optional): Mains frequency. Defaults to 60.
            model_type (str, optional): Model type to use, can be CNN or MLP. Defaults to "CNN".
            negative_method (str, optional): Method to handle negative labels. Can be "mixed" or "none". Defaults to "mixed".
            relabel_method (str, optional): Relabelling method. Can be "LabelSpreading" or "none". Defaults to "none".
            gesture_ids (Iterable, optional): List of gesture IDs. Defaults to (1, 2, 3, 4, 5, 8, 26, 30).
        """
        self.subject_id = subject_id
        self.sensor = EmgSensor(
            sensor_type,
            notch_freq=powerline_notch_freq,
            window_size_ms=200,
            window_inc_ms=50,
            majority_vote_ms=0,
        )
        self.stage = stage

        self.finetune = finetune
        self.model_type = model_type
        self.negative_method = negative_method
        self.relabel_method = relabel_method
        self.gesture_ids = gesture_ids

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

        # The steps before game are common so we can work in the no_adap dir first, and then copy stuff when >= game
        if self.stage >= ExperimentStage.GAME and self.adaptation:
            src = self.paths.get_experiment_dir()
            self.paths.trial = "adap"

            dest = self.paths.get_experiment_dir()
            if "adap" not in os.listdir(self.paths.base):
                os.mkdir(dest)

            # Just always override the common files for safety...
            shutil.copy(src + "model.pth", dest + "model.pth")
            shutil.copy(src + "results_pre.json", dest + "results_pre.json")
            shutil.copytree(src + "train/", dest + "train/", dirs_exist_ok=True)
            shutil.copytree(src + "pre_test/", dest + "pre_test/", dirs_exist_ok=True)

        # Set testing data path
        if self.stage < ExperimentStage.SG_POST_TEST:
            self.paths.test = self.paths.test.replace("test", "pre_test")
        else:
            self.paths.test = self.paths.test.replace("test", "post_test")

        # Set results file path
        if self.stage == ExperimentStage.SG_PRE_TEST:
            self.paths.results = "results_pre.json"
        elif self.stage == ExperimentStage.GAME:
            self.paths.results = "results_live.csv"
        elif self.stage == ExperimentStage.SG_POST_TEST:
            self.paths.results = "results_post.json"

    def get_feature_parameters(self):
        if isinstance(self.features, str):
            fe = libemg.feature_extractor.FeatureExtractor()
            self.features = fe.get_feature_groups()[self.features]

    def get_datacollection_parameters(self):
        self.reps = 5 if not self.finetune else 1
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
        torch.set_float32_matmul_precision("high")

        # Load or create model
        if self.stage >= ExperimentStage.SG_PRE_TEST or self.finetune:
            # Load post-game model
            if self.stage == ExperimentStage.SG_POST_TEST:
                self.paths.set_model("model_post")

            log.info(f"Loading model from {self.paths.get_model()}")

            if self.model_type == "CNN":
                self.model = models.load_conv(
                    self.paths.get_model(), self.num_channels, self.input_shape
                )
            elif self.model_type == "MLP":
                self.model = models.load_mlp(self.paths.get_model())
            else:
                raise ValueError("Invalid model type.")
        elif (
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

        # Only finetune last layer except for the game stage
        if self.finetune and self.stage == ExperimentStage.SG_TRAIN:
            log.warning("========== Enabling finetuning ==========")
            self.model.feature_extractor.requires_grad_(False)
        else:
            log.info("Model NOT in finetuning.")
            self.model.feature_extractor.requires_grad_(True)

        self.model.to(self.accelerator)

    def get_game_parameters(self):
        self.game_time = 600
