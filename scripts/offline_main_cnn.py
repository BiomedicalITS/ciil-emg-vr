import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

from libemg.feature_extractor import FeatureExtractor

from nfc_emg import utils
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths
from nfc_emg.models import EmgCNN, main_train_nn, main_test_nn, get_model_cnn

import configs as g


def __main():

    SENSOR = EmgSensorType.BioArmband

    SAMPLE_DATA = False
    SAMPLE_TEST_DATA = False
    FINETUNE = False

    GESTURE_IDS = g.FUNCTIONAL_SET

    sensor = EmgSensor(SENSOR)
    paths = NfcPaths("data/" + sensor.get_name(), 1)
    paths.set_model_name("model_cnn")

    fe = FeatureExtractor()
    fg = fe.get_feature_groups()["HTD"]

    model = EmgCNN(len(fg), sensor.emg_shape, len(GESTURE_IDS))
    # model = get_model(paths.model, sensor.emg_shape, len(GESTURE_IDS), FINETUNE)
    model = main_train_nn(
        model=model,
        sensor=sensor,
        sample_data=SAMPLE_DATA,
        features=fg,
        gestures_list=GESTURE_IDS,
        gestures_dir=paths.gestures,
        data_dir=paths.train,
        model_out_path=paths.model,
    )

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
