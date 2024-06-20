from nfc_emg.sensors import EmgSensorType, EmgSensor
from nfc_emg.utils import get_online_data_handler

if __name__ == "__main__":
    sensor = EmgSensor(EmgSensorType.BioArmband)
    sensor.start_streamer()
    odh = get_online_data_handler(
        sensor.fs,
        imu=False,
        attach_filters=(
            False if sensor.sensor_type == EmgSensorType.BioArmband else True
        ),
    )
    odh.visualize_channels(list(range(8)), 6000)
