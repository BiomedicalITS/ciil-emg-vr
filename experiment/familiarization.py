import numpy as np

from nfc_emg.utils import get_online_data_handler

from config import Config


class Familiarization:
    def __init__(self, config: Config):
        self.config = config

    def run(self):
        self.config.sensor.start_streamer()
        self.odh = get_online_data_handler(self.config.sensor, False)

        # self.odh.visualize_channels(
        #     list(range(np.prod(self.config.sensor.emg_shape))),
        #     3 * self.config.sensor.fs,
        # )

        self.odh.visualize(3 * self.config.sensor.fs)
