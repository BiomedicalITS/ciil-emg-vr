import numpy as np
import torch

from emager_py.majority_vote import majority_vote

from libemg.offline_metrics import OfflineMetrics

from nfc_emg import utils, datasets, models
from nfc_emg.models import EmgCNN
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths


def main_cnn(
    sensor: EmgSensor,
    sample_data: bool,
    gestures_list: list,
    gestures_dir: str,
    data_dir: str,
    finetune: bool,
    model_out_path: str,
):
    if sample_data:
        sensor.start_streamer()
        odh = utils.get_online_data_handler(
            sensor.fs, notch_freq=sensor.notch_freq, imu=False
        )
        utils.screen_guided_training(
            odh, gestures_list, gestures_dir, 1 if finetune else 5, 5, data_dir
        )

    classes = utils.get_cid_from_gid(gestures_dir, data_dir, gestures_list)
    reps = utils.get_reps(data_dir)
    if len(reps) == 1:
        train_reps = reps
        test_reps = []
    else:
        train_reps = reps[: int(0.8 * len(reps))]
        test_reps = reps[int(0.8 * len(reps)) :]

    print("Training on reps:", train_reps)
    print("Testing on reps:", test_reps)

    model = (
        EmgCNN(sensor.emg_shape, len(gestures_list))
        if not finetune
        else models.get_model(
            model_out_path, sensor.emg_shape, len(gestures_list), True
        )
    )
    model = models.train_cnn(model, sensor, data_dir, classes, train_reps, test_reps)
    torch.save(model.state_dict(), model_out_path)
    return model


def main_test_scnn(
    model: EmgCNN,
    sensor: EmgSensor,
    data_dir: str,
    sample_data: bool,
    gestures_list: list,
    gestures_dir: str,
):
    if sample_data:
        sensor.start_streamer()
        odh = utils.get_online_data_handler(
            sensor.fs,
            sensor.bandpass_freqs,
            sensor.notch_freq,
            False,
            False if sensor.sensor_type == EmgSensorType.BioArmband else True,
        )
        utils.screen_guided_training(odh, gestures_list, gestures_dir, 1, 5, data_dir)

    classes = utils.get_cid_from_gid(gestures_dir, data_dir, gestures_list)
    reps = utils.get_reps(data_dir)
    idle_cid = utils.map_gid_to_cid(gestures_dir, data_dir)[1]

    preds, labels = models.test_model(model, sensor, data_dir, classes, reps)
    preds_maj = majority_vote(preds, sensor.maj_vote_n)

    om = OfflineMetrics()
    metrics = ["CA", "AER", "INS", "REJ_RATE", "CONF_MAT", "RECALL", "PREC", "F1"]
    results = om.extract_offline_metrics(metrics, labels, preds, idle_cid)
    results_maj = om.extract_offline_metrics(metrics, labels, preds_maj, idle_cid)

    print(f"Precision RAW: {results['PREC']}")
    print(f"Precision MAJ: {results_maj['PREC']}")

    return results_maj


def __main():
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    import configs as g

    SENSOR = EmgSensorType.MyoArmband

    SAMPLE_DATA = False
    SAMPLE_TEST_DATA = False
    FINETUNE = False

    # TRAIN_GESTURE_IDS = [1, 2, 3, 4, 5, 8, 14, 26, 30]
    TRAIN_GESTURE_IDS = [1, 2, 3, 8, 14, 26, 30]

    sensor = EmgSensor(SENSOR)
    sensor.set_majority_vote(200)
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

    model = models.get_model(
        paths.model, sensor.emg_shape, len(TRAIN_GESTURE_IDS), FINETUNE
    )
    # model = main_cnn(
    #     sensor=sensor,
    #     sample_data=SAMPLE_DATA,
    #     gestures_list=TRAIN_GESTURE_IDS,
    #     gestures_dir=paths.gestures,
    #     data_dir=paths.train if not FINETUNE else paths.fine,
    #     finetune=FINETUNE,
    #     model_out_path=paths.model,
    # )
    model.to(g.ACCELERATOR)

    test_results = main_test_scnn(
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
