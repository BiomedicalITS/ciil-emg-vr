import numpy as np

from nfc_emg import utils
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths
from nfc_emg.models import main_cnn, main_test_cnn


def __main():
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    import configs as g

    SENSOR = EmgSensorType.BioArmband

    SAMPLE_DATA = False
    SAMPLE_TEST_DATA = False
    FINETUNE = False

    # TRAIN_GESTURE_IDS = [1, 2, 3, 4, 5, 8, 14, 26, 30]
    TRAIN_GESTURE_IDS = [1, 2, 3, 8, 14, 26, 30]

    sensor = EmgSensor(SENSOR)
    paths = NfcPaths("data/" + sensor.get_name())
    paths.model = paths.model.replace("model", "model_cnn")

    print("*" * 80)
    print(
        "Chosen gestures:",
        utils.map_gid_to_name(paths.gestures, TRAIN_GESTURE_IDS),
    )
    print(
        "GIDs to CIDs:",
        utils.map_cid_to_ordered_name(paths.gestures, paths.train, TRAIN_GESTURE_IDS),
    )
    print("*" * 80)

    # model = models.get_model(
    #     paths.model, sensor.emg_shape, len(TRAIN_GESTURE_IDS), FINETUNE
    # )
    model = main_cnn(
        sensor=sensor,
        sample_data=SAMPLE_DATA,
        gestures_list=TRAIN_GESTURE_IDS,
        gestures_dir=paths.gestures,
        data_dir=paths.train if not FINETUNE else paths.fine,
        finetune=FINETUNE,
        model_out_path=paths.model,
    )
    model.to(g.ACCELERATOR)

    test_results = main_test_cnn(
        model=model,
        sensor=sensor,
        data_dir=paths.test,
        sample_data=SAMPLE_TEST_DATA,
        gestures_list=TRAIN_GESTURE_IDS,
        gestures_dir=paths.gestures,
    )
    conf_mat = test_results["CONF_MAT"] / np.sum(
        test_results["CONF_MAT"], axis=1, keepdims=True
    )
    test_gesture_names = utils.get_name_from_gid(
        paths.gestures, paths.train, TRAIN_GESTURE_IDS
    )
    ConfusionMatrixDisplay(conf_mat, display_labels=test_gesture_names).plot()
    plt.show()


if __name__ == "__main__":
    __main()
