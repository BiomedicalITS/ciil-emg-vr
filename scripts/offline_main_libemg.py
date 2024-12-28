import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.metrics import ConfusionMatrixDisplay

from libemg.emg_classifier import EMGClassifier
from libemg.offline_metrics import OfflineMetrics
from libemg.feature_extractor import FeatureExtractor

from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths

from nfc_emg import utils, datasets

import configs as g


def __main():
    SAMPLE_DATA = False
    # SAMPLE_DATA = True

    # FEATURE_SETS = ["TDPSD", "MSWT", "LS4", "TDAR"]
    FEATURE_SETS = ["TDPSD"]
    GESTURE_IDS = g.FUNCTIONAL_SET
    SENSOR = EmgSensorType.BioArmband

    sensor = EmgSensor(SENSOR, window_size_ms=150, window_inc_ms=50, majority_vote_ms=0)
    # paths = NfcPaths(f"data/{sensor.get_name()}", -1)
    paths = NfcPaths(f"data/0/{sensor.get_name()}", "no_adap")
    paths.gestures = "data/gestures/"

    train_dir = paths.get_train()
    paths.test = "pre_test/"
    test_dir = paths.get_test()

    if SAMPLE_DATA:
        utils.do_sgt(sensor, GESTURE_IDS, paths.gestures, train_dir, 5, 3)
    # elif paths.trial == paths.get_next_trial():
    #     paths.set_trial(paths.trial - 1)

    gestures_cids = utils.get_cid_from_gid(paths.gestures, train_dir, GESTURE_IDS)
    idle_id = utils.map_gid_to_cid(paths.gestures, train_dir)[1]
    reps = utils.get_reps(train_dir)

    train_odh = datasets.get_offline_datahandler(train_dir, gestures_cids, reps)
    test_odh = datasets.get_offline_datahandler(test_dir, gestures_cids, reps)

    # if sensor.sensor_type == EmgSensorType.BioArmband:
    #     fi = utils.get_filter(sensor.fs, sensor.bandpass_freqs, sensor.notch_freq)
    #     fi.filter(train_odh)
    #     fi.filter(test_odh)

    train_windows, train_labels = datasets.prepare_data(train_odh, sensor)
    test_windows, test_labels = datasets.prepare_data(test_odh, sensor)

    classifier = "MLP"
    print("=========================================")
    print(f"{classifier}")
    print("=========================================")

    for feature_set in FEATURE_SETS:
        print("=========================================")
        print(f"Feature set: {feature_set}")
        print("=========================================")

        data_set = {
            "training_features": FeatureExtractor().extract_feature_group(
                feature_set, train_windows
            ),
            "training_labels": train_labels,
        }

        test_features = FeatureExtractor().extract_feature_group(
            feature_set, test_windows
        )

        FeatureExtractor().visualize_feature_space(
            data_set["training_features"],
            projection="PCA",
            classes=train_labels,
            # test_feature_dic=test_features,
            # t_classes=test_labels,
            render=True,
        )

        for k, v in data_set["training_features"].items():
            scaler = preprocessing.StandardScaler()
            scaler.fit(v)
            data_set["training_features"][k] = scaler.transform(v)
            test_features[k] = scaler.transform(test_features[k])

        model = EMGClassifier()
        model.fit(classifier, data_set.copy())
        # model.add_majority_vote(sensor.maj_vote_n)
        preds, probs = model.run(test_features)

        om = OfflineMetrics()
        metrics = ["CA", "AER", "REJ_RATE", "CONF_MAT"]
        results = om.extract_offline_metrics(
            metrics, test_labels, preds, null_label=idle_id
        )
        for key in results:
            if key == "CONF_MAT":
                continue
            print(f"{key}: {results[key]}")

        conf_mat = results["CONF_MAT"] / np.sum(
            results["CONF_MAT"], axis=1, keepdims=True
        )
        test_gesture_names = utils.get_name_from_gid(
            paths.gestures, train_dir, GESTURE_IDS
        )
        ConfusionMatrixDisplay(conf_mat, display_labels=test_gesture_names).plot()
        plt.title(
            f"{sensor.get_name()} using an {classifier} with {feature_set} features"
        )

    plt.show()

    # conf_mat = test_results["CONF_MAT"] / np.sum(
    #     test_results["CONF_MAT"], axis=1, keepdims=True
    # )
    # test_gesture_names = utils.get_name_from_gid(
    #     paths.gestures, paths.test, GESTURE_IDS
    # )

    # ConfusionMatrixDisplay(conf_mat, display_labels=test_gesture_names).plot()
    # plt.show()


if __name__ == "__main__":
    __main()
