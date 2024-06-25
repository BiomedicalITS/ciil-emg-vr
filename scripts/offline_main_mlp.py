import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np
import torch

from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier
from libemg.feature_extractor import FeatureExtractor

from nfc_emg import utils
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths
from nfc_emg import models
from nfc_emg.models import EmgMLP, main_train_nn, main_test_nn

import configs as g


def __main():

    SENSOR = EmgSensorType.BioArmband

    SAMPLE_DATA = False
    SAMPLE_TEST_DATA = False
    FINETUNE = False

    GESTURE_IDS = g.FUNCTIONAL_SET

    sensor = EmgSensor(SENSOR)
    paths = NfcPaths("data/" + sensor.get_name(), 0)
    paths.set_model_name("model_mlp")

    fe = FeatureExtractor()
    fg = fe.get_feature_groups()["HTD"]

    model = models.load_mlp(paths.model)
    # model = EmgMLP(len(fg) * np.prod(sensor.emg_shape), len(GESTURE_IDS))
    # model = main_train_nn(
    #     model=model,
    #     sensor=sensor,
    #     sample_data=SAMPLE_DATA,
    #     features=fg,
    #     gestures_list=GESTURE_IDS,
    #     gestures_dir=paths.gestures,
    #     data_dir=paths.train,
    #     model_out_path=paths.model,
    # )

    classi = EMGClassifier()
    classi.add_majority_vote(sensor.maj_vote_n)
    classi.classifier = model.eval()

    sensor.start_streamer()
    odh = utils.get_online_data_handler(sensor.fs, sensor.bandpass_freqs, imu=False, attach_filters=False if sensor.sensor_type == EmgSensorType.BioArmband else True)
    
    oclassi = OnlineEMGClassifier(classi, sensor.window_size, sensor.window_increment, odh, fg, port=g.PREDS_PORT, std_out=True)
    oclassi.run()

    test_results = main_test_nn(
        model=model,
        sensor=sensor,
        features=fg,
        data_dir=paths.test,
        sample_data=SAMPLE_TEST_DATA,
        gestures_list=GESTURE_IDS,
        gestures_dir=paths.gestures,
    )
    conf_mat = test_results["CONF_MAT"] / np.sum(
        test_results["CONF_MAT"], axis=1, keepdims=True
    )
    test_gesture_names = utils.get_name_from_gid(
        paths.gestures, paths.train, GESTURE_IDS
    )
    ConfusionMatrixDisplay(conf_mat, display_labels=test_gesture_names).plot()
    plt.show()


if __name__ == "__main__":
    __main()
