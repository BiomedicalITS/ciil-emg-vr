from nfc_emg.utils import get_online_data_handler, map_cid_to_ordered_name

from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier

from config import Config


class Familiarization:
    def __init__(self, config: Config, classification: bool):
        self.config = config
        self.classification = classification

    def run(self):
        self.config.sensor.start_streamer()
        self.odh = get_online_data_handler(self.config.sensor, False)

        # self.odh.visualize_channels(
        #     list(range(np.prod(self.config.sensor.emg_shape))),
        #     3 * self.config.sensor.fs,
        # )
        if self.classification:
            classi = EMGClassifier()
            classi.classifier = self.config.model.to("cuda").eval()
            classi.add_majority_vote(self.config.sensor.maj_vote_n)
            ws, wi = self.config.sensor.window_size, self.config.sensor.window_increment

            print("=" * 100)
            print(
                map_cid_to_ordered_name(
                    self.config.paths.gestures,
                    self.config.paths.get_train(),
                    self.config.gesture_ids,
                )
            )
            print("=" * 100)

            oclassi = OnlineEMGClassifier(
                classi, ws, wi, self.odh, self.config.features, port=12347, std_out=True
            )
            oclassi.run(block=True)
        else:
            self.odh.visualize(3 * self.config.sensor.fs)
