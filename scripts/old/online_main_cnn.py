import socket
import time
import logging as log
from tqdm import tqdm
from pyquaternion import Quaternion

from emager_py import majority_vote
from emager_py.data_processing import cosine_similarity

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from scipy.special import softmax

from nfc_emg import utils, models, schemas as s, datasets
import configs as g


class OnlineDataWrapper:
    def __init__(
        self,
        sensor: str,
        model_path: str,
        emg_fs: int,
        emg_shape: tuple,
        emg_buffer_size: int,
        emg_window_size_ms: int,
        emg_maj_vote_ms: int,
        gestures_num: int,
        gestures_dir: str,
        accelerator: str,
        use_imu: bool,
        pseudo_labels_port: int = 5111,
        preds_ip: str = "127.0.0.1",
        preds_port: int = 5112,
    ):
        """
        Main object to do the NFC-EMG stuff.

        Starts listening to the data stream.

        Params:
            - sensor: str, the sensor used. "emager", "myo" or "bio"
            - emg_fs: int, the EMG sampling rate
            - emg_shape: tuple, the shape of the EMG data
            - emg_buffer_size: int, the size of the EMG buffer to keep in memory
            - emg_window_size: int, minimum size of new EMG data to process in ms
            - emg_maj_vote_ms: int, the majority vote window in ms
            - num_gestures: int, the number of gestures to classify
            - accelerator: str, the device to run the model on ("cuda", "mps", "cpu", etc)
        """

        self.sensor = sensor
        self.accelerator = accelerator
        self.use_imu = use_imu

        utils.setup_streamer(self.sensor)
        self.odh = utils.get_online_data_handler(
            emg_fs, notch_freq=50, imu=use_imu, max_buffer=emg_buffer_size
        )

        # I/O sockets
        self.pl_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pl_socket.bind(("127.0.0.1", pseudo_labels_port))
        self.pl_socket.setblocking(False)

        self.preds_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.preds_sock = (preds_ip, preds_port)

        # Models
        self.model = models.get_model_cnn(model_path, emg_shape, gestures_num, True)
        self.model.to(self.accelerator)
        self.optimizer = self.model.configure_optimizers()

        # EMG params
        self.emg_shape = emg_shape
        self.emg_buffer_size = emg_buffer_size
        self.emg_window_size = emg_window_size_ms * emg_fs // 1000
        self.last_emg_sample = np.zeros((1, *self.emg_shape))
        self.emg_buffer = np.zeros((emg_buffer_size, *self.emg_shape))

        self.voter = majority_vote.MajorityVote(emg_maj_vote_ms * emg_fs // 1000)
        self.gesture_dict = utils.map_cid_to_name(gestures_dir)

        # IMU params
        if self.use_imu:
            self.imu_window_size = 10 if self.sensor == "myo" else 20  # 200 ms
            self.arm_movement_list = s.ArmControl._member_names_
            self.arm_calib_data = np.zeros((len(self.arm_movement_list), 4))
            self.arm_calib_deadzones = [0] * len(self.arm_movement_list)
            self.calibrate_imu()
            self.save_calibration_data()

    def calibrate_imu(self, n_samples=100):
        """
        Ask to move the arm in 3d space. For each "axis" covered, take the mean quaternion.

        Thus, for each arm movement we obtain a triplet of pitch, yaw, roll. Save it.

        Then cosine similarity to find the most approprite axis.
        """

        ans = input("Do you want to calibrate the IMU? (y/N): ")
        if ans != "y":
            self.load_calibration_data()
            return

        if self.sensor == "bio":
            ans = input("Is the Armband coming out of deep sleep? (y/N): ")
            if ans == "y":
                input(
                    "Put the armband on a stable surface to calibrate its IMU. Press Enter when ready."
                )
                calib_samples = 250
                fetched_samples = 0
                with tqdm(total=calib_samples) as pbar:
                    while fetched_samples < calib_samples:
                        quats = self.get_imu_data("quat")
                        if quats is None:
                            time.sleep(0.1)
                            continue
                        fetched_samples += len(quats)
                        pbar.update(len(quats))

        for i, v in enumerate(self.arm_movement_list):
            input(
                f"({i+1}/{len(self.arm_movement_list)}) Move the arm to the {v.upper()} position and press Enter."
            )
            self.odh.raw_data.reset_imu()
            fetched_samples = 0
            with tqdm(total=n_samples) as pbar:
                while fetched_samples < n_samples:
                    quats = self.get_imu_data("quat")
                    if quats is None:
                        time.sleep(0.1)
                        continue
                    fetched_samples += len(quats)
                    pbar.update(len(quats))
                    quats = np.sum(quats, axis=0)
                    self.arm_calib_data[i] += quats
            self.arm_calib_data[i] = self.arm_calib_data[i] / fetched_samples
        q_neutral = Quaternion(self.arm_calib_data[0])
        for i in range(1, len(self.arm_movement_list)):
            q_pos = Quaternion(self.arm_calib_data[i])
            self.arm_calib_deadzones[i] = (
                Quaternion.absolute_distance(q_neutral, q_pos) / 2
            )

        log.info(f"Calibration quats:\n{self.arm_calib_data}")
        log.info(f"Deadzones:\n{self.arm_calib_deadzones}")

    def save_calibration_data(self):
        np.savez(
            f"data/{self.sensor}/imu_calib_data.npz",
            arm_calib_data=self.arm_calib_data,
            arm_calib_deadzones=np.array(self.arm_calib_deadzones),
        )

    def load_calibration_data(self):
        data = np.load(f"data/{self.sensor}/imu_calib_data.npz")
        self.arm_calib_data = data["arm_calib_data"]
        self.arm_calib_deadzones = list(data["arm_calib_deadzones"])
        log.info(f"Loaded calibration data:\n {self.arm_calib_data}")
        log.info(f"Loaded calibration deadzones: {self.arm_calib_deadzones}")

    def run(self):
        while True:
            t0 = time.perf_counter()

            quats = self.get_imu_data("quat", self.imu_window_size)
            imu_command = self.get_arm_movement(quats)

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
                losses = self.finetune_model(
                    self.emg_buffer.reshape((-1, 1, *self.emg_shape)),
                    np.repeat(labels, self.emg_buffer_size),
                )
                print(
                    f"Finetuned on label {labels.item()}. Average loss: {sum(losses)/len(losses)}"
                )

            if preds is not None:
                self.voter.extend(preds)
                maj_vote = self.voter.vote().item(0)
                self.send_pred(maj_vote)
                print("*" * 80)
                print("EMG length:", len(emg_data))
                print(f"Gesture {self.gesture_dict[maj_vote]} ({maj_vote})")
                print(f"Time taken {time.perf_counter() - t0:.4f} s")

    def get_imu_data(self, channel: str, sample_windows: int = 1):
        """
        Get IMU quaternions, return the attitude readings. Clear the odh buffer.

        Params:
            - channel: str, "quat", "acc" or "both"

        If channel is "quat", the quaternions are normalized.
        """

        start, end = 0, 4
        if channel == "acc":
            start, end = 4, 7
        elif channel == "both":
            start, end = 0, 7

        odata = self.odh.get_imu_data()  # (n_samples, n_ch)
        if len(odata) < sample_windows:
            return None
        self.odh.raw_data.reset_imu()
        if self.sensor == "bio":
            # myo is (quat, acc, gyro)
            # bio is (acc, quat)
            odata = np.roll(odata, 4, axis=1)
        # log.info(f"IMU data shape: {odata.shape}")
        vals = odata[:, start:end]
        if channel == "quat":
            vals = vals / np.linalg.norm(vals, axis=1, keepdims=True)
        return vals

    def get_arm_movement(self, quats: np.ndarray):
        """
        TODO: maybe consider the degree of similarity instead of just the closest

        Get the movement from quaternions.

        Params:
            - quats: np.ndarray, shape (n, 4): The quaternions to process.

        Returns the arm position. None if quats is empty.
        """
        if quats is None:
            return None

        quats = np.mean(quats, axis=0, keepdims=True)
        sim_score = cosine_similarity(quats, self.arm_calib_data, False)
        id = np.argmax(sim_score, axis=1).item()
        distance = Quaternion.absolute_distance(
            Quaternion(quats.T), Quaternion(self.arm_calib_data[0].T)
        )
        # log.info(f"Similarity matrix:\n {sim_score}")
        # log.info(f"Distance: {distance}")
        if distance < self.arm_calib_deadzones[id]:
            # Inside of dead zone so assume neutral
            return s.ArmControl.NEUTRAL
        return s.ArmControl[self.arm_movement_list[id]]

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
            data = datasets.process_data(data, self.sensor, sample_windows)
        return data

    def predict_from_emg(self, data: np.ndarray):
        """
        Run the model on the given data.

        Returns None if data is empty. Else, an array of shape [n_samples,]
        TODO show confidence%
        """
        if data is None:
            return None
        data = data.reshape(-1, 1, *self.emg_shape)
        samples = torch.from_numpy(data).to(self.accelerator)
        pred = self.model(samples).cpu().detach().numpy()
        pred = softmax(pred, axis=1)
        preds = np.mean(pred, axis=0)
        vals = [f"{p:.3f}" for p in preds]
        print(f"Predictions:\n {vals}")
        # pred[pred < 0.7] = 0
        return np.argmax(pred, axis=1)

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
    # from emager_py.utils import set_logging

    # set_logging()

    odw = OnlineDataWrapper(
        sensor=g.DEVICE,
        model_path=g.MODEL_PATH,
        emg_fs=g.EMG_SAMPLING_RATE,
        emg_shape=g.EMG_DATA_SHAPE,
        emg_buffer_size=g.EMG_SAMPLING_RATE,
        emg_window_size_ms=25,
        emg_maj_vote_ms=g.EMG_MAJ_VOTE_MS,
        gestures_num=len(g.LIBEMG_GESTURE_IDS),
        gestures_dir=g.TRAIN_DATA_DIR,
        accelerator=g.ACCELERATOR,
        use_imu=True,
        pseudo_labels_port=g.PSEUDO_LABELS_PORT,
        preds_ip=g.PREDS_IP,
        preds_port=g.PREDS_PORT,
    )
    # odw.visualize_emg()
    odw.run()
    while True:
        t0 = time.perf_counter()
        print("*" * 80)
        print(f"Time taken {time.perf_counter() - t0:.4f} s")
        time.sleep(0.1)
