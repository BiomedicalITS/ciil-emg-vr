import numpy as np

from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths
from nfc_emg.models import main_train_scnn, main_finetune_scnn, main_test_scnn
from nfc_emg import models, utils


def __main():
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from nfc_emg.models import CosineSimilarity
    import configs as g

    SENSOR = EmgSensorType.BioArmband
    SAMPLE_DATA = False
    SAMPLE_FINE_DATA = False
    SAMPLE_TEST_DATA = False

    GESTURE_IDS = [1, 2, 3, 4, 5, 17, 18]

    sensor = EmgSensor(SENSOR)
    paths = NfcPaths(f"data/{sensor.get_name()}_wrist")

    try:
        print("*" * 80)
        print(
            "Chosen gestures:",
            utils.map_gid_to_name(paths.gestures, GESTURE_IDS),
        )
        print(
            "Ordered CIDs:",
            utils.map_cid_to_ordered_name(paths.gestures, paths.train, GESTURE_IDS),
        )
        print("*" * 80)
    except Exception:
        pass

    mw = models.get_scnn(paths.model, sensor.emg_shape)
    # mw = main_train_scnn(
    #     sensor=sensor,
    #     data_dir=paths.train,
    #     sample_data=SAMPLE_DATA,
    #     gestures_list=GESTURE_IDS,
    #     gestures_dir=paths.gestures,
    #     # classifier=CosineSimilarity(),
    #     classifier=LinearDiscriminantAnalysis(),
    #     model_out_path=paths.model,
    # )
    # mw.attach_classifier(LinearDiscriminantAnalysis())

    main_finetune_scnn(
        mw=mw,
        sensor=sensor,
        data_dir=paths.fine,
        sample_data=SAMPLE_FINE_DATA,
        gestures_list=GESTURE_IDS,
        gestures_dir=paths.gestures,
    )
    # models.save_scnn(mw, paths.model)

    test_results = main_test_scnn(
        mw=mw,
        sensor=sensor,
        data_dir=paths.test,
        sample_data=SAMPLE_TEST_DATA,
        gestures_list=GESTURE_IDS,
        gestures_dir=paths.gestures,
    )

    conf_mat = test_results["CONF_MAT"] / np.sum(
        test_results["CONF_MAT"], axis=1, keepdims=True
    )
    test_gesture_names = utils.get_name_from_gid(
        paths.gestures, paths.test, GESTURE_IDS
    )

    ConfusionMatrixDisplay(conf_mat, display_labels=test_gesture_names).plot()
    plt.show()


if __name__ == "__main__":
    __main()
