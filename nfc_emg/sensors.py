from enum import Enum

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
        window_size_ms: int = 150,
        window_inc_ms: int = 20,
        majority_vote_ms: int = 200,
    ):
        self.sensor_type = sensor_type
        if sensor_type == EmgSensorType.BioArmband:
            self.fs = 2000
            self.emg_shape = (8,)
        elif sensor_type == EmgSensorType.MyoArmband:
            self.fs = 200
            self.emg_shape = (8,)
        elif sensor_type == EmgSensorType.Emager:
            self.fs = 1000
            self.emg_shape = (4, 16)

        self.notch_freq = notch_freq
        self.bandpass_freqs = bandpass_freqs
        self.set_window_size(window_size_ms)
        self.set_window_increment(window_inc_ms)
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
            return self.p
        elif self.sensor_type == EmgSensorType.MyoArmband:
            self.p = myo_streamer(filtered=False, imu=True)
        elif self.sensor_type == EmgSensorType.BioArmband:
            self.p = sifibridge_streamer(
                device="BioArmband",
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

    def stop_streamer(self):
        """Stop the streamer for the device"""
        if self.p is not None:
            self.p.terminate()
            self.p = None

    def set_window_size(self, ms: int):
        if ms == 0:
            self.window_size = 1
        else:
            self.window_size = (ms * self.fs) // 1000

    def set_window_increment(self, ms: int):
        """
        Set window increment. If ms is 0, set to 1 sample.
        """
        if ms == 0:
            self.window_increment = 1
        else:
            self.window_increment = (ms * self.fs) // 1000

    def set_majority_vote(self, ms: int):
        if ms == 0:
            self.maj_vote_n = 1
        else:
            self.maj_vote_n = (ms * self.fs) // (1000 * self.window_increment)
