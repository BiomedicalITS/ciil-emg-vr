import socket
import time

from emager_py import majority_vote
from emager_py.data_processing import cosine_similarity

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nfc_emg import utils, models, datasets
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.models import EmgSCNN
import configs as g


class OnlineDataWrapper:
    def __init__(
        self,
        sensor: EmgSensor,
        model: EmgSCNN,
        gestures_id_list: list,
        gestures_dir: str,
        dataset_dir: str,
        accelerator: str,
        pseudo_labels_port: int = 5111,
        preds_ip: str = "127.0.0.1",
        preds_port: int = 5112,
    ):
        """
        Main object to do the NFC-EMG stuff.

        Starts listening to the data stream.

        Params:
            - gestures_id_list: list of LibEMG Gestures IDs to classify
            - accelerator: str, the device to run the model on ("cuda", "mps", "cpu", etc)
        """

        self.sensor = sensor

        self.sensor.start_streamer()

        self.odh = utils.get_online_data_handler(
            sensor.fs,
            sensor.bandpass_freqs,
            sensor.notch_freq,
            True,
            max_buffer=sensor.fs,
        )

        self.pl_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pl_socket.bind(("127.0.0.1", pseudo_labels_port))
        self.pl_socket.setblocking(False)

        self.preds_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.preds_sock = (preds_ip, preds_port)

        self.accelerator = accelerator

        self.model = model
        self.model.to(self.accelerator)
        self.model.eval()
        # print("Classifier classes:", self.model.classifier.classes_)

        self.emg_buffer_size = self.sensor.fs
        self.emg_buffer = np.zeros(
            (self.emg_buffer_size, *self.sensor.emg_shape), np.float32
        )

        self.voter = majority_vote.MajorityVote(self.sensor.maj_vote_n)

        # Convert GIDs to CIDs
        self.class_names = [
            utils.map_gid_to_name(gestures_dir)[gid] for gid in gestures_id_list
        ]

    def run(self):
        while True:
            t0 = time.perf_counter()

            emg_data = self.get_emg_data(True, self.sensor.moving_avg_n)
            preds = self.predict(emg_data)

            if emg_data is not None:
                new_data = emg_data
                if len(emg_data) > len(self.emg_buffer):
                    new_data = emg_data[-len(self.emg_buffer) :]
                self.emg_buffer = np.roll(self.emg_buffer, -len(new_data), axis=0)
                self.emg_buffer[-len(new_data) :] = new_data

            labels = self.get_live_labels()
            if labels is not None:
                self.model.fit_classifier(
                    self.emg_buffer, np.repeat(labels, len(self.emg_buffer))
                )
                print(f"Calibrated on label {labels.item()}.")

            if preds is not None:
                self.voter.extend(preds)
                maj_vote = self.voter.vote().item(0)
                self.send_pred(maj_vote)
                print("*" * 80)
                print("EMG length:", len(emg_data))
                print(f"Gesture {self.class_names[maj_vote]} ({maj_vote})")
                print(f"Time taken {time.perf_counter() - t0:.4f} s")

    def get_emg_data(self, process: bool, sample_windows: int):
        """Get EMG data.

        Returns None if no new data. Otherwise the data with shape (n_samples, *emg_shape)
        """
        odata = self.odh.get_data()
        if len(odata) < sample_windows:
            return None
        self.odh.raw_data.reset_emg()

        data = np.reshape(odata, (-1, *self.sensor.emg_shape))
        if process:
            data = datasets.process_data(data, self.sensor)
        return data

    def predict(self, data: np.ndarray):
        """Infer embeddings from the given data.

        Args:
            data (np.ndarray): Data with shape (n_samples, H, W)

        Returns:
            _type_: _description_
        """
        if data is None:
            return None
        return model.predict(data)

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

    def finetune_model(
        self,
        data: np.ndarray,
        labels: np.ndarray,
        epochs: int = 10,
    ):
        """
        Finetune the model on the given data and labels.
        """
        data = torch.from_numpy(data.astype(np.float32)).to(self.accelerator)
        labels = torch.from_numpy(labels.astype(np.uint8)).to(self.accelerator)
        dataloader = DataLoader(TensorDataset(data, labels), batch_size=64)
        self.model.train()
        losses = []
        for _ in range(epochs):
            for data, labels in dataloader:
                loss = self.model.training_step((data, labels), 0)
                loss.backward()
                self.optimizer.step()
                self.optimizer.zero_grad()
                losses.append(loss.item())
        self.model.eval()
        return losses

    def visualize_emg(self):
        self.odh.visualize()


if __name__ == "__main__":
    import time

    sensor = EmgSensor(EmgSensorType.MyoArmband)
    dataset_dir, _, model_path, gestures_dir = utils.set_paths(sensor.get_name())
    model = models.get_model_scnn(model_path, sensor.emg_shape)

    odw = OnlineDataWrapper(
        sensor,
        model,
        [1, 2, 3, 26, 30],
        gestures_dir,
        dataset_dir,
        g.ACCELERATOR,
        g.PEUDO_LABELS_PORT,
        g.PREDS_IP,
        g.PREDS_PORT,
    )
    # odw.visualize_emg()
    odw.run()
