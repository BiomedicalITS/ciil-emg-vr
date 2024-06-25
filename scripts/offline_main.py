import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths
from nfc_emg.models import (
    EmgSCNNWrapper,
    CosineSimilarity,
    main_train_scnn,
    main_finetune_scnn,
    main_test_scnn,
)
from nfc_emg import models, utils

import configs as g


def __main():
    SENSOR = EmgSensorType.BioArmband
    GESTURE_IDS = g.FUNCTIONAL_SET

    SAMPLE_DATA = False
    SAMPLE_FINE_DATA = False
    SAMPLE_TEST_DATA = False

    # SAMPLE_DATA = True
    # SAMPLE_FINE_DATA = True
    # SAMPLE_TEST_DATA = True

    sensor = EmgSensor(SENSOR, window_inc_ms=10)
    paths = NfcPaths(f"data/{sensor.get_name()}", 2)

    sensor.set_window_increment(5)

    if paths.trial_number == paths.get_next_trial():
        paths.set_trial_number(
            paths.trial_number if SAMPLE_DATA else paths.trial_number - 1
        )

    # mw = EmgSCNNWrapper.load_from_disk(paths.model, sensor.emg_shape, 1, "cpu")
    mw = main_train_scnn(
        sensor=sensor,
        data_dir=paths.train,
        sample_data=SAMPLE_DATA,
        gestures_list=GESTURE_IDS,
        gestures_dir=paths.gestures,
        # classifier=CosineSimilarity(),
        classifier=LinearDiscriminantAnalysis(),
    )
    mw.save_to_disk(paths.model)

    mw.attach_classifier(LinearDiscriminantAnalysis())
    # mw.attach_classifier(CosineSimilarity())

    main_finetune_scnn(
        mw=mw,
        sensor=sensor,
        data_dir=paths.fine,
        sample_data=SAMPLE_FINE_DATA,
        gestures_list=GESTURE_IDS,
        gestures_dir=paths.gestures,
    )

    test_results = main_test_scnn(
        mw=mw,
        sensor=sensor,
        data_dir=paths.test,
        sample_data=SAMPLE_TEST_DATA,
        gestures_list=GESTURE_IDS,
        gestures_dir=paths.gestures,
    )

    utils.save_eval_results(test_results, paths.results)

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
