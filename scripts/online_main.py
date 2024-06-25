import socket
import time
import json

import numpy as np

from libemg.feature_extractor import FeatureExtractor
from libemg.data_handler import get_windows

from emager_py import majority_vote

from nfc_emg import utils, models, datasets, schemas as s
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.models import EmgCNN, EmgSCNNWrapper, EmgMLP
from nfc_emg.paths import NfcPaths


class OnlineDataWrapper:
    def __init__(
        self,
        sensor: EmgSensor,
        mw: EmgSCNNWrapper | EmgCNN,
        features: list,
        paths: NfcPaths,
        gestures_id_list: list,
        pseudo_labels_port: int = 5111,
        preds_ip: str = "127.0.0.1",
        preds_port: int = 5112,
    ):
        """
        Main object to do the NFC-EMG stuff.

        Starts listening to the data stream.

        Params:
            - base_dir: base directory of the experiment
            - gestures_id_list: list of LibEMG Gestures IDs to classify
            - accelerator: str, the device to run the model on ("cuda", "mps", "cpu", etc)
        """

        self.sensor = sensor
        self.paths = paths
        self.odh = utils.get_online_data_handler(
            sensor.fs,
            sensor.bandpass_freqs,
            sensor.notch_freq,
            False,
            False if sensor.sensor_type == EmgSensorType.BioArmband else True,
            max_buffer=sensor.fs,
        )

        self.sensor.start_streamer()

        self.fe = FeatureExtractor()
        self.fg = features

        self.pl_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pl_socket.bind(("127.0.0.1", pseudo_labels_port))
        self.pl_socket.setblocking(False)

        self.preds_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.preds_sock = (preds_ip, preds_port)

        self.mw = mw

        self.emg_buffer_size = self.sensor.fs
        self.emg_buffer = np.zeros(
            (self.emg_buffer_size, np.prod(self.sensor.emg_shape)), np.float32
        )

        self.voter = majority_vote.MajorityVote(self.sensor.maj_vote_n)

        # Convert GIDs to CIDs
        self.class_names = utils.get_name_from_gid(
            paths.gestures, paths.train, gestures_id_list
        )
        self.name_to_cid = utils.reverse_dict(utils.map_cid_to_name(paths.train))

    def run(self, timeout=60):
        start = time.time()
        while time.time() - start < timeout:
            t0 = time.perf_counter()

            emg_data = self.get_emg_data()
            preds = self.predict(emg_data)

            if emg_data is not None:
                new_data = emg_data
                if len(emg_data) > len(self.emg_buffer):
                    new_data = emg_data[-len(self.emg_buffer) :]
                self.emg_buffer = np.roll(self.emg_buffer, -len(new_data), axis=0)
                self.emg_buffer[-len(new_data) :] = new_data

            labels = self.get_live_labels()
            if labels is not None:
                item = s.ObjectShape.get_possible_gestures(
                    self.name_to_cid, labels.item()
                )
                probas = self.mw.predict_proba(self.emg_buffer)

                self.mw.fit(self.emg_buffer, np.repeat(labels, len(self.emg_buffer)))
                print(f"Calibrated on label {labels.item()}.")

            if preds is not None:
                self.voter.extend(preds)
                maj_vote = self.voter.vote().item(0)
                self.send_pred(maj_vote)
                print("*" * 80)
                print("EMG length:", len(emg_data))
                print(f"Gesture {self.class_names[maj_vote]} ({maj_vote})")
                print(f"Time taken {time.perf_counter() - t0:.4f} s")

        print(f"Experiment finished after {timeout} s. Exiting.")

    def get_emg_data(self):
        """Get EMG data.

        Returns None if no new data. Otherwise the data with shape (n_samples, *emg_shape)
        """
        odata = self.odh.get_data()
        if len(odata) < self.sensor.window_size:
            return None
        self.odh.raw_data.reset_emg()
        return odata

    def predict(self, data: np.ndarray):
        """Infer embeddings from the given data.

        Args:
            data (np.ndarray): Data with shape (n_samples, H, W)

        Returns:
            _type_: _description_
        """
        if data is None:
            return None
        windows = get_windows(data, self.sensor.window_size, self.sensor.window_increment)
        features = self.fe.extract_features(self.fg, windows, array=True)
        return self.mw.predict(features)

    def get_live_labels(self):
        """
        Run the socket and return the data.
        """
        try:
            sockdata = np.frombuffer(self.pl_socket.recv(2048), dtype=np.uint8)
            return sockdata
        except Exception:
            return None

    def send_pred(self, label: int):
        """
        Send the robot command to the robot.
        """
        self.preds_socket.sendto(bytes([label]), self.preds_sock)

    def visualize_emg(self):
        self.odh.visualize()


def __main():
    import configs as g
    import torch

    SENSOR = EmgSensorType.BioArmband
    GESTURE_IDS = g.FUNCTIONAL_SET

    sensor = EmgSensor(SENSOR)
    paths = NfcPaths(f"data/{sensor.get_name()}")

    paths.set_trial_number(paths.trial_number - 1)
    paths.set_model_name("model_mlp")

    fe = FeatureExtractor()
    fg = fe.get_feature_groups()["HTD"]

    model = EmgMLP(len(fg) * np.prod(sensor.emg_shape), len(GESTURE_IDS))
    model.load_state_dict(torch.load(paths.model))
    model.eval()

    # models.main_finetune_scnn(
    #     mw, sensor, paths.fine, False, GESTURE_IDS, paths.gestures
    # )
    # models.save_scnn(mw, paths.model)

    odw = OnlineDataWrapper(
        sensor,
        model,
        fg,
        paths,
        GESTURE_IDS,
        g.PSEUDO_LABELS_PORT,
        g.PREDS_IP,
        g.PREDS_PORT,
    )
    # odw.visualize_emg()
    odw.run()


if __name__ == "__main__":
    __main()
