import torch
import numpy as np
from sklearn.base import BaseEstimator
from sklearn.metrics import accuracy_score

from libemg.emg_classifier import EMGClassifier
from libemg.offline_metrics import OfflineMetrics

from nfc_emg import utils, datasets, models
from nfc_emg.models import EmgSCNN, EmgCNN
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths


def train_scnn(
    sensor: EmgSensor,
    data_dir: str,
    sample_data: bool,
    gestures_list: list,
    gestures_dir: str,
    classifier: BaseEstimator,
    model_out_path: str,
    accelerator: str,
):
    if sample_data:
        sensor.start_streamer()
        odh = utils.get_online_data_handler(
            sensor.fs, sensor.bandpass_freqs, sensor.notch_freq, False
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

    model = EmgSCNN(sensor.emg_shape, classifier).to(accelerator)
    model = models.train_model_scnn(
        model, sensor, data_dir, classes, train_reps, test_reps
    )
    models.save_model_scnn(model, model_out_path)
    return model


def test_scnn(
    model: EmgSCNN,
    sensor: EmgSensor,
    data_dir: str,
    sample_data: bool,
    gestures_list: list,
    gestures_dir: str,
):
    if sample_data:
        sensor.start_streamer()
        odh = utils.get_online_data_handler(
            sensor.fs, sensor.bandpass_freqs, sensor.notch_freq, False
        )
        utils.screen_guided_training(odh, gestures_list, gestures_dir, 1, 5, data_dir)

    classes = utils.get_cid_from_gid(gestures_dir, data_dir, gestures_list)
    reps = utils.get_reps(data_dir)
    reps = reps[int(0.8 * len(reps)) :]

    odh = datasets.get_offline_datahandler(data_dir, classes, reps)
    test_data, test_labels = datasets.prepare_data(odh, sensor, 1, 1)

    model.eval()
    offc = EMGClassifier()
    offc.classifier = model
    offc.add_majority_vote(sensor.maj_vote_n)
    preds = offc.run(test_data)

    om = OfflineMetrics()
    metrics = ["CA", "AER", "INS", "REJ_RATE", "CONF_MAT", "RECALL", "PREC", "F1"]
    results = om.extract_offline_metrics(metrics, test_labels, preds[0], null_label=2)
    for key in results:
        print(f"{key}: {results[key]}")
    return results


def main_cnn(
    sensor: EmgSensor,
    sample_data: bool,
    gestures_list: list,
    gestures_img_dir: str,
    out_data_dir: str,
    finetune: bool,
    model_out_path: str,
):
    if sample_data:
        sensor.start_streamer()
        odh = utils.get_online_data_handler(
            sensor.fs, notch_freq=sensor.notch_freq, imu=False
        )
        utils.screen_guided_training(
            odh, gestures_list, gestures_img_dir, 1 if finetune else 5, 5, out_data_dir
        )

    reps = utils.get_reps(out_data_dir)
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
    model = models.train_model(model, sensor, out_data_dir, train_reps, test_reps)
    torch.save(model.state_dict(), model_out_path)
    return model


def __main():
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from nfc_emg.models import CosineSimilarity
    import configs as g

    SENSOR = EmgSensorType.BioArmband
    SAMPLE_DATA = False
    SAMPLE_TEST_DATA = False

    TRAIN_GESTURE_IDS = [1, 2, 3, 4, 5, 8, 14, 26, 30]
    TEST_GESTURE_IDS = [1, 2, 3, 4, 5, 8, 14, 26, 30]

    sensor = EmgSensor(SENSOR)
    sensor.set_majority_vote(200)

    paths = NfcPaths("data/" + sensor.get_name())

    # model = models.get_model_scnn(paths.model, sensor.emg_shape, g.ACCELERATOR)
    model = train_scnn(
        sensor=sensor,
        data_dir=paths.train,
        sample_data=SAMPLE_DATA,
        gestures_list=TRAIN_GESTURE_IDS,
        gestures_dir=paths.gestures,
        # classifier=CosineSimilarity(),
        classifier=LinearDiscriminantAnalysis(),
        model_out_path=paths.model,
        accelerator=g.ACCELERATOR,
    )

    test_results = test_scnn(
        model=model,
        sensor=sensor,
        data_dir=paths.test,
        sample_data=SAMPLE_TEST_DATA,
        gestures_list=TEST_GESTURE_IDS,
        gestures_dir=paths.gestures,
    )
    conf_mat = test_results["CONF_MAT"] / np.linalg.norm(
        test_results["CONF_MAT"], axis=1, keepdims=True
    )
    test_gesture_names = [
        utils.map_gid_to_name(paths.gestures)[i] for i in TEST_GESTURE_IDS
    ]
    ConfusionMatrixDisplay(conf_mat, display_labels=test_gesture_names).plot()
    plt.show()


if __name__ == "__main__":
    __main()
