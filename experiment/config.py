from enum import IntEnum

import libemg

from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths
from nfc_emg import models


class ExperimentStage(IntEnum):
    FAMILIARIZATION = 0
    SG_TRAIN = 1
    SG_TEST = 2
    GAME = 3
    POST_SG_TEST = 4


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
        self.sensor = EmgSensor(sensor_type)

        self.adaptation = adaptation
        self.negative_method = "mixed"
        self.relabel_method = "LabelSpreading"

        self.gesture_ids = [1, 2, 3, 4, 5, 8, 26, 30]
        self.features = "TDPSD"  # Can be list of features OR feature group

        self.get_feature_parameters()
        self.get_path_parameters()
        self.get_datacollection_parameters()
        self.get_classifier_parameters()
        self.get_game_parameters()

    def get_path_parameters(self):
        self.paths = NfcPaths(f"data/{self.subject_id}/{self.sensor.get_name()}/")
        if self.stage > ExperimentStage.SG_TRAIN:
            self.paths.set_trial_number(self.paths.trial_number - 1)
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

        if self.stage <= ExperimentStage.SG_TRAIN:
            # Create a new model
            self.model = models.EmgCNN(
                len(self.features), self.sensor.emg_shape, len(self.gesture_ids)
            )
            return
        elif self.stage >= ExperimentStage.POST_SG_TEST:
            self.paths.set_model_name("model_post")

        # Load model from disk
        self.model = models.load_conv(
            self.paths.model, self.num_channels, self.input_shape
        )

    def get_game_parameters(self):
        self.game_time = 600
