import numpy as np
from sklearn.preprocessing import Normalizer
import torch

from sklearn.utils import shuffle
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from emager_py.majority_vote import majority_vote

from libemg.emg_classifier import EMGClassifier
from libemg.offline_metrics import OfflineMetrics

from nfc_emg import utils, datasets, models
from nfc_emg.models import EmgSCNN, EmgCNN, EmgSCNNWrapper
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths


def main_train_scnn(
    sensor: EmgSensor,
    data_dir: str,
    sample_data: bool,
    gestures_list: list,
    gestures_dir: str,
    classifier: BaseEstimator,
    model_out_path: str,
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
        utils.screen_guided_training(odh, gestures_list, gestures_dir, 5, 5, data_dir)

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

    model = EmgSCNN(sensor.emg_shape)
    mw = EmgSCNNWrapper(model, classifier)
    mw = models.train_scnn(mw, sensor, data_dir, classes, train_reps, test_reps)
    models.save_scnn(mw, model_out_path)
    return mw


def main_finetune_scnn(
    mw: EmgSCNNWrapper,
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

    odh = datasets.get_offline_datahandler(data_dir, classes, reps)
    data, labels = datasets.prepare_data(odh, sensor, 1, 1)

    train_data, test_data, train_labels, test_labels = train_test_split(
        data, labels, test_size=0.2, shuffle=True
    )
    train_data, train_labels = shuffle(train_data, train_labels)

    # Fit classifier
    mw.model.eval()
    mw.fit(train_data, train_labels)

    # Test
    preds = mw.predict(test_data)

    om = OfflineMetrics()
    metrics = ["CA", "AER", "INS", "REJ_RATE", "CONF_MAT", "RECALL", "PREC", "F1"]
    results = om.extract_offline_metrics(
        metrics, test_labels, preds, null_label=gestures_list.index(1)
    )
    for key in results:
        print(f"{key}: {results[key]}")
    return mw


def main_test_scnn(
    mw: EmgSCNNWrapper,
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

    odh = datasets.get_offline_datahandler(data_dir, classes, reps)
    test_data, test_labels = datasets.prepare_data(odh, sensor, 1, 1)
    idle_id = utils.map_gid_to_cid(gestures_dir, data_dir)[1]

    mw.model.eval()
    preds = mw.predict(test_data)
    preds_maj = majority_vote(preds, sensor.maj_vote_n)

    for i in range(len(set(test_labels))):
        print(set(preds[test_labels == i]))
    print(set(preds))

    acc = accuracy_score(test_labels, preds)
    print(acc)

    om = OfflineMetrics()
    metrics = ["CA", "AER", "INS", "REJ_RATE", "CONF_MAT", "RECALL", "PREC", "F1"]

    results = om.extract_offline_metrics(
        metrics, test_labels, preds, null_label=idle_id
    )
    results_maj = om.extract_offline_metrics(
        metrics, test_labels, preds_maj, null_label=idle_id
    )
    print(f"Precision RAW: {results['PREC']}")
    print(f"Precision MAJ: {results_maj['PREC']}")
    return results_maj


def __main():
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from nfc_emg.models import CosineSimilarity
    import configs as g

    SENSOR = EmgSensorType.BioArmband
    SAMPLE_DATA = False
    SAMPLE_TEST_DATA = False

    # TRAIN_GESTURE_IDS = [1, 2, 3, 4, 5, 8, 14, 26, 30]
    TRAIN_GESTURE_IDS = [1, 2, 3, 8, 14, 26, 30]
    TEST_GESTURE_IDS = [1, 2, 3, 8, 14, 26, 30]

    sensor = EmgSensor(SENSOR)
    paths = NfcPaths("data/" + sensor.get_name())

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

    mw = models.get_scnn(paths.model, sensor.emg_shape)
    # mw = main_train_scnn(
    #     sensor=sensor,
    #     data_dir=paths.train,
    #     sample_data=SAMPLE_DATA,
    #     gestures_list=TRAIN_GESTURE_IDS,
    #     gestures_dir=paths.gestures,
    #     # classifier=CosineSimilarity(),
    #     classifier=LinearDiscriminantAnalysis(),
    #     model_out_path=paths.model,
    # )
    mw.model.to(g.ACCELERATOR)

    main_finetune_scnn(
        mw=mw,
        sensor=sensor,
        data_dir=paths.fine,
        sample_data=SAMPLE_DATA,
        gestures_list=TRAIN_GESTURE_IDS,
        gestures_dir=paths.gestures,
    )

    test_results = main_test_scnn(
        mw=mw,
        sensor=sensor,
        data_dir=paths.test,
        sample_data=SAMPLE_TEST_DATA,
        gestures_list=TEST_GESTURE_IDS,
        gestures_dir=paths.gestures,
    )

    conf_mat = test_results["CONF_MAT"] / np.sum(
        test_results["CONF_MAT"], axis=1, keepdims=True
    )
    test_gesture_names = utils.get_name_from_gid(
        paths.gestures, paths.test, TEST_GESTURE_IDS
    )

    ConfusionMatrixDisplay(conf_mat, display_labels=test_gesture_names).plot()
    plt.show()


if __name__ == "__main__":
    __main()
