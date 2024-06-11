import socket
import time

from emager_py import majority_vote
from emager_py.data_processing import cosine_similarity

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from nfc_emg import utils
import configs as g


class OnlineDataWrapper:
    def __init__(
        self,
        device: str,
        emg_fs: int,
        emg_shape: tuple,
        emg_buffer_size: int,
        emg_window_size_ms: int,
        emg_maj_vote_ms: int,
        num_gestures: int,
        accelerator: str,
        pseudo_labels_port: int = 5111,
        preds_ip: str = "127.0.0.1",
        preds_port: int = 5112,
    ):
        """
        Main object to do the NFC-EMG stuff.

        Starts listening to the data stream.

        Params:
            - device: str, the sensor used. "emager", "myo" or "bio"
            - emg_fs: int, the EMG sampling rate
            - emg_shape: tuple, the shape of the EMG data
            - emg_buffer_size: int, the size of the EMG buffer to keep in memory
            - emg_window_size: int, minimum size of new EMG data to process in ms
            - emg_maj_vote_ms: int, the majority vote window in ms
            - num_gestures: int, the number of gestures to classify
            - accelerator: str, the device to run the model on ("cuda", "mps", "cpu", etc)
        """

        self.device = device
        utils.setup_streamer(self.device)
        self.odh = utils.get_online_data_handler(
            emg_fs, notch_freq=50, imu=False, max_buffer=emg_buffer_size
        )

        self.pl_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pl_socket.bind(("127.0.0.1", pseudo_labels_port))
        self.pl_socket.setblocking(False)

        self.preds_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.preds_sock = (preds_ip, preds_port)

        self.accelerator = accelerator

        self.model = utils.get_model(g.MODEL_PATH, emg_shape, num_gestures, True, True)
        self.model.to(self.accelerator)
        self.model.eval()
        self.model.embeddings = self.model.embeddings / np.linalg.norm(
            self.model.embeddings, axis=1, keepdims=True
        )
        self.optimizer = self.model.configure_optimizers()

        self.emg_shape = emg_shape
        self.emg_buffer_size = emg_buffer_size
        self.emg_window_size = emg_window_size_ms * emg_fs // 1000
        self.last_emg_sample = np.zeros((1, *self.emg_shape))
        self.emg_buffer = np.zeros((emg_buffer_size, *self.emg_shape), np.float32)

        self.voter = majority_vote.MajorityVote(emg_maj_vote_ms * emg_fs // 1000)
        self.gesture_dict = utils.map_cid_to_name(g.TRAIN_DATA_DIR)

    def run(self):
        while True:
            t0 = time.perf_counter()

            emg_data = self.get_emg_data(True, self.emg_window_size)
            preds = self.predict_from_emg(emg_data)

            if emg_data is not None:
                new_data = emg_data
                if len(emg_data) > len(self.emg_buffer):
                    new_data = emg_data[-len(self.emg_buffer) :]
                self.emg_buffer = np.roll(self.emg_buffer, -len(new_data), axis=0)
                self.emg_buffer[-len(new_data) :] = new_data

            labels = self.get_live_labels()
            if labels is not None:
                embeds = self.infer_from_data(self.emg_buffer)
                embeds /= np.linalg.norm(embeds, axis=1, keepdims=True)
                centroid = np.mean(embeds, axis=0)
                new_centroid = (centroid + self.model.embeddings[labels]) / 2
                self.model.embeddings[labels] = new_centroid
                print(f"Calibrated on label {labels.item()}.")

            if preds is not None:
                self.voter.extend(preds)
                maj_vote = self.voter.vote().item(0)
                self.send_pred(maj_vote)
                print("*" * 80)
                print("EMG length:", len(emg_data))
                print(f"Gesture {self.gesture_dict[maj_vote]} ({maj_vote})")
                print(f"Time taken {time.perf_counter() - t0:.4f} s")

    def get_emg_data(self, process: bool, sample_windows: int):
        """Get EMG data.

        Returns None if no new data. Otherwise the data with shape (n_samples, *emg_shape)
        """
        odata = self.odh.get_data()
        if len(odata) < sample_windows:
            return None
        self.odh.raw_data.reset_emg()

        data = np.reshape(odata, (-1, *self.emg_shape))
        if process:
            data = utils.process_data(data)
        return data

    def infer_from_data(self, data: np.ndarray):
        """Infer embeddings from the given data.

        Args:
            data (np.ndarray): Data with shape (n_samples, *emg_shape)

        Returns:
            _type_: _description_
        """
        if data is None:
            return None
        data = data.reshape(-1, 1, *self.emg_shape)
        samples = torch.from_numpy(data).to(self.accelerator)
        embeddings = self.model(samples).cpu().detach().numpy()
        return embeddings

    def predict_from_emg(self, data: np.ndarray):
        """
        Run the model on the given data.

        Returns None if data is empty. Else, an array of shape [n_samples,]
        TODO show confidence%
        """
        if data is None:
            return None
        embeddings = self.infer_from_data(data)
        y = cosine_similarity(embeddings, self.model.embeddings, False)
        return np.argmax(y, axis=1)

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

    odw = OnlineDataWrapper(
        g.DEVICE,
        g.EMG_SAMPLING_RATE,
        g.EMG_DATA_SHAPE,
        g.EMG_SAMPLING_RATE,
        25,
        g.EMG_MAJ_VOTE_MS,
        len(g.LIBEMG_GESTURE_IDS),
        g.ACCELERATOR,
        g.PEUDO_LABELS_PORT,
        g.PREDS_IP,
        g.PREDS_PORT,
    )
    # odw.visualize_emg()
    odw.run()
