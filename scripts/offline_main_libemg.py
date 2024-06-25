import numpy as np
from sklearn import preprocessing

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

    FEATURE_SETS = ["HTD"]
    GESTURE_IDS = g.FUNCTIONAL_SET
    SENSOR = EmgSensorType.BioArmband

    sensor = EmgSensor(SENSOR)
    paths = NfcPaths(f"data/{sensor.get_name()}", 0)

    if SAMPLE_DATA:
        utils.do_sgt(sensor, GESTURE_IDS, paths.gestures, paths.train, 5, 3)
    elif paths.trial_number == paths.get_next_trial():
        paths.set_trial_number(paths.trial_number - 1)

    fe = FeatureExtractor()
    om = OfflineMetrics()

    train_odh = datasets.get_offline_datahandler(paths.train, GESTURE_IDS, [0, 1, 2])
    test_odh = datasets.get_offline_datahandler(paths.test, GESTURE_IDS, [0])

    train_windows, train_labels = datasets.prepare_data(train_odh, sensor)
    test_windows, test_labels = datasets.prepare_data(test_odh, sensor)

    for feature_set in FEATURE_SETS:
        print("=========================================")
        print(f"Feature set: {feature_set}")
        print("=========================================")
        data_set = {}
        data_set["training_features"] = fe.extract_feature_group(
            feature_set, train_windows
        )
        data_set["training_labels"] = train_labels
        test_features = fe.extract_feature_group(feature_set, test_windows)
        idle_id = utils.map_gid_to_cid(paths.gestures, paths.train)[1]

        for k, v in data_set["training_features"].items():
            scaler = preprocessing.StandardScaler()
            scaler.fit(v)
            data_set["training_features"][k] = scaler.transform(v)
            test_features[k] = scaler.transform(test_features[k])
            print(np.mean(data_set["training_features"][k], axis=0))

        classifiers = ["LDA", "SVM", "QDA"]
        for classifier in classifiers:
            model = EMGClassifier()
            model.fit(classifier, data_set.copy())
            model.add_majority_vote(sensor.maj_vote_n)
            preds, probs = model.run(test_features)
            metrics = om.extract_common_metrics(test_labels, preds, idle_id)
            print(f"--- {classifier} ---")
            print(metrics)

    fe.visualize_feature_space(
        data_set["training_features"],
        projection="PCA",
        classes=train_labels,
        test_feature_dic=test_features,
        t_classes=test_labels,
    )

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
