import time

from libemg.feature_extractor import FeatureExtractor
from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier

from nfc_emg.sensors import EmgSensorType, EmgSensor
from nfc_emg.utils import get_online_data_handler
from nfc_emg import datasets, utils, models
from nfc_emg.paths import NfcPaths

import configs as g

def test_oclassi():
    SENSOR = EmgSensorType.BioArmband

    sensor = EmgSensor(SENSOR)
    sensor.start_streamer()

    paths = NfcPaths(f"data/vr_{sensor.get_name()}")
    paths.set_trial_number(paths.trial_number - 1)
    paths.set_model_name("model_mlp")
    paths.gestures = "data/gestures/"

    model = models.load_mlp(paths.model)
    model.__setattr__("classes_", list(range(len(g.FUNCTIONAL_SET))))

    odh = utils.get_online_data_handler(sensor.fs, sensor.bandpass_freqs, sensor.notch_freq, False, False if SENSOR == EmgSensorType.BioArmband else True)

    classi = EMGClassifier()
    classi.add_majority_vote(sensor.maj_vote_n)
    classi.classifier = model.eval()

    oclassi = OnlineEMGClassifier(classi, sensor.window_size, sensor.window_increment, odh, g.FEATURES, std_out=False, output_format="probabilities")
    oclassi.run(block=False)
    time.sleep(5)
    oclassi.analyze_classifier()

def test_rand():
    sensor = EmgSensor(EmgSensorType.BioArmband)
    paths = NfcPaths(f"data/{sensor.get_name()}")
    paths.set_trial_number(paths.trial_number - 1)

    classes = utils.map_gid_to_cid(paths.gestures, paths.train)

    # sensor.start_streamer()

    # odh = get_online_data_handler(
    #     sensor.fs,
    #     imu=False,
    #     attach_filters=(
    #         False if sensor.sensor_type == EmgSensorType.BioArmband else True
    #     ),
    # )
    # odh.visualize_channels(list(range(8)), 6000)

    odh = datasets.get_offline_datahandler(
        paths.train, list(classes.values()), utils.get_reps(paths.train)
    )
    windows, labels = datasets.prepare_data(odh, sensor)

    fe = FeatureExtractor()
    features = fe.extract_feature_group("HTD", windows)
    print("Allo")

if __name__ == "__main__":
    test_oclassi()    
