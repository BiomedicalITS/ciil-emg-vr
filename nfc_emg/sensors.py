from enum import Enum
import numpy as np

from libemg.streamers import sifibridge_streamer, myo_streamer, emager_streamer


class EmgSensorType(Enum):
    BioArmband = "bio"
    MyoArmband = "myo"
    Emager = "emager"


class EmgSensor:
    def __init__(
        self,
        sensor_type: EmgSensorType,
        notch_freq: int = 50,
        bandpass_freqs: tuple = (20, 450),
        moving_average_ms: int = 25,
        majority_vote_ms: int = 200,
    ):
        self.sensor_type = sensor_type
        if sensor_type == EmgSensorType.BioArmband:
            self.fs = 1500
            self.emg_shape = (1, 8)
        elif sensor_type == EmgSensorType.MyoArmband:
            self.fs = 200
            self.emg_shape = (1, 8)
        elif sensor_type == EmgSensorType.Emager:
            self.fs = 1000
            self.emg_shape = (4, 16)

        self.notch_freq = notch_freq
        self.bandpass_freqs = bandpass_freqs
        self.set_moving_average(moving_average_ms)
        self.set_majority_vote(majority_vote_ms)
        self.p = None

    def get_name(self):
        return self.sensor_type.value

    def start_streamer(self):
        """Setup the streamer for the device

        Returns:
            process handle
        """
        if self.p is not None:
            pass
        elif self.sensor_type == EmgSensorType.MyoArmband:
            self.p = myo_streamer(filtered=False, imu=True)
        elif self.sensor_type == EmgSensorType.BioArmband:
            self.p = sifibridge_streamer(
                version="1_1",
                emg=True,
                imu=True,
                notch_on=True,
                notch_freq=self.notch_freq,
                emg_fir_on=True,
                emg_fir=self.bandpass_freqs,
            )
        elif self.sensor_type == EmgSensorType.Emager:
            self.p = emager_streamer()

        return self.p

    def reorder(self, data: np.ndarray):
        """Reorder EMG data.

        Args:
            data (np.ndarray): 3 or 4D array of EMG data (N[, 1], H, W)
        """
        if self.sensor_type == EmgSensorType.BioArmband:
            return data[..., [4, 6, 3, 0, 7, 1, 2, 5]]
        else:
            return data

    def set_majority_vote(self, ms: int):
        self.maj_vote_n = (ms * self.fs) // 1000

    def set_moving_average(self, ms: int):
        self.moving_avg_n = (ms * self.fs) // 1000
