import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

from libemg.emg_classifier import EMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.offline_metrics import OfflineMetrics

from nfc_emg import utils, datasets
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths

import configs as g


def __main():
    # SAMPLE_DATA = True
    # SAMPLE_TEST_DATA = True
    # FINETUNE = True

    sensor = EmgSensor(
        g.SENSOR, window_size_ms=150, window_inc_ms=20, majority_vote_ms=0
    )

    paths = NfcPaths(f"data/{sensor.get_name()}", -1)
    paths.gestures = "data/gestures/"

    train_dir = paths.get_train()
    test_dir = paths.get_test()

    classes = utils.get_cid_from_gid(paths.gestures, train_dir, g.FUNCTIONAL_SET)
    idle_cid = utils.map_gid_to_cid(paths.gestures, train_dir)[1]

    scaler = StandardScaler()

    train_odh = datasets.get_offline_datahandler(
        train_dir, classes, utils.get_reps(train_dir)
    )
    train_win, train_labels = datasets.prepare_data(train_odh, sensor)
    train_data = FeatureExtractor().extract_features(g.FEATURES, train_win, array=True)
    train_data = scaler.fit_transform(train_data)

    test_odh = datasets.get_offline_datahandler(
        test_dir, classes, utils.get_reps(test_dir)
    )
    test_win, test_labels = datasets.prepare_data(test_odh, sensor)
    test_data = FeatureExtractor().extract_features(g.FEATURES, test_win, array=True)
    test_data = scaler.transform(test_data)

    # model = LinearDiscriminantAnalysis()
    model = MLPClassifier()
    model.fit(train_data, train_labels)

    if isinstance(model, MLPClassifier):
        print(f"Number of inputs:  {model.n_features_in_}")
        print(f"Number of outputs: {model.n_outputs_}")
        print(f"Number of layers:  {model.n_layers_}")
        print(f"Layer sizes: {[layer.shape for layer in model.coefs_]}")

    classi = EMGClassifier()
    # classi.add_majority_vote(sensor.maj_vote_n)
    # classi.add_rejection(0.9)
    classi.classifier = model

    test_preds, _ = classi.run(test_data)

    om = OfflineMetrics()
    metrics = ["CA", "AER", "REJ_RATE", "CONF_MAT"]
    results = om.extract_offline_metrics(
        metrics, test_labels, test_preds, null_label=idle_cid
    )

    for key in results:
        if key == "CONF_MAT":
            continue
        print(f"{key}: {results[key]}")

    conf_mat = results["CONF_MAT"] / np.sum(results["CONF_MAT"], axis=1, keepdims=True)
    test_gesture_names = utils.get_name_from_gid(
        paths.gestures, train_dir, g.FUNCTIONAL_SET
    )
    ConfusionMatrixDisplay(conf_mat, display_labels=test_gesture_names).plot()
    plt.show()

    # sensor.start_streamer()
    # odh = utils.get_online_data_handler(sensor.fs, sensor.bandpass_freqs, imu=False, attach_filters=False if sensor.sensor_type == EmgSensorType.BioArmband else True)
    # classi = EMGClassifier()
    # classi.add_majority_vote(sensor.maj_vote_n)
    # classi.classifier = model.eval()
    # oclassi = OnlineEMGClassifier(classi, sensor.window_size, sensor.window_increment, odh, fg, port=g.PREDS_PORT, std_out=True)
    # oclassi.run()


if __name__ == "__main__":
    __main()
