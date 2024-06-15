import socket
import time
import json

import numpy as np
from pyquaternion import Quaternion
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from tqdm import tqdm

from emager_py import majority_vote
from emager_py.data_processing import cosine_similarity

from nfc_emg import utils, models, datasets, schemas as s
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.models import EmgCNN, EmgSCNNWrapper
from nfc_emg.paths import NfcPaths


class OnlineDataWrapper:
    def __init__(
        self,
        sensor: EmgSensor,
        mw: EmgSCNNWrapper | EmgCNN,
        paths: NfcPaths,
        gestures_id_list: list,
        accelerator: str,
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
            True,
            max_buffer=sensor.fs,
        )

        self.sensor.start_streamer()

        self.pl_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pl_socket.bind(("127.0.0.1", pseudo_labels_port))
        self.pl_socket.setblocking(False)

        self.preds_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.preds_sock = (preds_ip, preds_port)

        self.accelerator = accelerator

        self.mw = mw

        self.emg_buffer_size = self.sensor.fs
        self.emg_buffer = np.zeros(
            (self.emg_buffer_size, *self.sensor.emg_shape), np.float32
        )

        self.voter = majority_vote.MajorityVote(self.sensor.maj_vote_n)

        # Convert GIDs to CIDs
        self.class_names = utils.get_name_from_gid(
            paths.gestures, paths.train, gestures_id_list
        )

        self.arm_movement_list = s.ArmControl._member_names_
        self.arm_calib_data = np.zeros((len(self.arm_movement_list), 4))
        self.arm_calib_deadzones = [0] * len(self.arm_movement_list)
        self.calibrate_imu()
        self.save_calibration_data()

    def run(self, timeout=60):
        last_arm_cmd = None
        start = time.time()
        while time.time() - start < timeout:
            t0 = time.perf_counter()

            quats = self.get_imu_data("quat", 5)
            imu_command = self.get_arm_movement(quats)

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

            # Send commands for arm, wrist, gripper
            if imu_command is not None and imu_command != last_arm_cmd:
                # print(f"Arm: {imu_command.name}")
                last_arm_cmd = imu_command
                self.send_robot_command(s.to_dict(imu_command))

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
        if self.sensor.sensor_type == EmgSensorType.BioArmband:
            # myo is (quat, acc, gyro)
            # bio is (acc, quat)
            odata = np.roll(odata, 4, axis=1)
        vals = odata[:, start:end]
        if channel == "quat":
            vals = vals / np.linalg.norm(vals, axis=1, keepdims=True)
        return vals

    def get_arm_movement(self, quats: np.ndarray):
        """Get the movement from quaternions.

        Params:
            - quats: np.ndarray, shape (n, 4): The quaternions to process.

        Returns the arm position. None if quats is empty.
        """
        if quats is None:
            return None

        quats = np.mean(quats, axis=0, keepdims=True)
        sim_score = cosine_similarity(quats, self.arm_calib_data, False)
        id = np.argmax(sim_score, axis=1).item(0)
        distance = Quaternion.absolute_distance(
            Quaternion(quats.T), Quaternion(self.arm_calib_data[0].T)
        )
        if distance < self.arm_calib_deadzones[id]:
            # Inside of dead zone so assume neutral
            return s.ArmControl[self.arm_movement_list[0]]
        return s.ArmControl[self.arm_movement_list[id]]

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
        return self.mw.predict(data)

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

    def send_robot_command(self, cmd: dict):
        """
        Send the robot command to the robot.
        """
        self.preds_socket.sendto(json.dumps(cmd).encode(), self.preds_sock)

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

        if self.sensor.sensor_type == EmgSensorType.BioArmband:
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
        # Accumulate data
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

        # Calculate deadzones
        for i in range(1, len(self.arm_movement_list)):
            q_pos = Quaternion(self.arm_calib_data[i])
            self.arm_calib_deadzones[i] = (
                Quaternion.absolute_distance(q_neutral, q_pos) / 2
            )

    def save_calibration_data(self):
        np.savez(
            self.paths.imu_calib,
            arm_calib_data=self.arm_calib_data,
            arm_calib_deadzones=np.array(self.arm_calib_deadzones),
        )

    def load_calibration_data(self):
        data = np.load(self.paths.imu_calib)
        self.arm_calib_data = data["arm_calib_data"]
        self.arm_calib_deadzones = list(data["arm_calib_deadzones"])

    def visualize_emg(self):
        self.odh.visualize()


def __main():
    import configs as g

    GESTURE_IDS = [1, 2, 3, 4, 5, 17, 18]
    SENSOR = EmgSensorType.BioArmband

    sensor = EmgSensor(SENSOR)

    paths = NfcPaths(f"data/{sensor.get_name()}_wrist")
    mw = models.get_scnn(paths.model, sensor.emg_shape)

    models.main_finetune_scnn(
        mw, sensor, paths.fine, False, GESTURE_IDS, paths.gestures
    )
    models.save_scnn(mw, paths.model)

    odw = OnlineDataWrapper(
        sensor,
        mw,
        paths,
        GESTURE_IDS,
        "cpu",
        g.PEUDO_LABELS_PORT,
        g.PREDS_IP,
        g.PREDS_PORT,
    )
    # odw.visualize_emg()
    odw.run()


if __name__ == "__main__":
    __main()
