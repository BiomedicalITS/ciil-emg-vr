from libemg.feature_extractor import FeatureExtractor

from nfc_emg.sensors import EmgSensorType, EmgSensor
from nfc_emg.utils import get_online_data_handler
from nfc_emg import datasets, utils
from nfc_emg.paths import NfcPaths

if __name__ == "__main__":
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
