import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier

from nfc_emg import utils
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths
from nfc_emg import models
from nfc_emg.models import EmgMLP, EmgConv1, main_train_nn, main_test_nn

import configs as g


def __main():
    SAMPLE_DATA = False
    SAMPLE_TEST_DATA = False
    FINETUNE = False

    # SAMPLE_DATA = True
    # SAMPLE_TEST_DATA = True
    # FINETUNE = True

    sensor = EmgSensor(g.SENSOR, majority_vote_ms=0)

    paths = NfcPaths(f"data/{sensor.get_name()}")
    if not SAMPLE_DATA or not SAMPLE_TEST_DATA:
        paths.set_trial_number(paths.get_next_trial() - 1)
        # paths.set_trial_number(-1)
    paths.gestures = "data/gestures/"

    print(f"Using data from trial {paths.trial_number}")

    # model = models.load_mlp(paths.model)
    # model = EmgMLP(len(g.FEATURES) * np.prod(sensor.emg_shape), len(g.FUNCTIONAL_SET))
    model = EmgConv1(len(g.FEATURES), np.prod(sensor.emg_shape), len(g.FUNCTIONAL_SET))

    model = main_train_nn(
        model=model,
        sensor=sensor,
        sample_data=SAMPLE_DATA,
        features=g.FEATURES,
        gestures_list=g.FUNCTIONAL_SET,
        gestures_dir=paths.gestures,
        data_dir=paths.fine if FINETUNE else paths.train,
        model_out_path=paths.model,
        num_reps=1 if FINETUNE else 5,
        rep_time=3
    )

    test_results = main_test_nn(
        model=model,
        sensor=sensor,
        features=g.FEATURES,
        data_dir=paths.test,
        sample_data=SAMPLE_TEST_DATA,
        gestures_list=g.FUNCTIONAL_SET,
        gestures_dir=paths.gestures,
    )

    conf_mat = test_results["CONF_MAT"] / np.sum(
        test_results["CONF_MAT"], axis=1, keepdims=True
    )
    test_gesture_names = utils.get_name_from_gid(
        paths.gestures, paths.train, g.FUNCTIONAL_SET
    )
    ConfusionMatrixDisplay(conf_mat, display_labels=test_gesture_names).plot()
    plt.show()

    # sensor.start_streamer()
    # odh = utils.get_online_data_handler(sensor.fs, sensor.bandpass_freqs, imu=False, attach_filters=False if sensor.sensor_type == EmgSensorType.BioArmband else True)
    # classi = EMGClassifier()
    # classi.add_majority_vote(sensor.maj_vote_n)
    # classi.classifier = model.eval()
    # oclassi = OnlineEMGClassifier(classi, sensor.window_size, sensor.window_increment, odh, g.FEATURES, std_out=True)
    # oclassi.run()

if __name__ == "__main__":
    __main()
