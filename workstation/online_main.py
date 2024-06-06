import vpython as vp
import logging as log
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import socket
import json
from tqdm import tqdm
import time
from pyquaternion import Quaternion

from emager_py import majority_vote
from emager_py.data_processing import cosine_similarity

import numpy as np
import torch
from scipy.special import softmax

from libemg.data_handler import OnlineDataHandler

import utils
import globals as g
import schemas as s


def get_attitude_from_quats(qw, qx, qy, qz):
    yaw = np.arctan2(2.0 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz)
    aasin = qx * qz - qw * qy
    pitch = np.arcsin(-2.0 * aasin)
    roll = np.arctan2(2.0 * (qx * qy + qw * qz), qw * qw + qx * qx - qy * qy - qz * qz)
    return pitch, yaw, roll


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
        robot_ip: str = "nvidia",
        robot_port: int = 5112,
        calibrate_imu: bool = True,
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
            emg_fs, notch_freq=50, max_buffer=emg_buffer_size
        )

        self.pl_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pl_socket.bind(("127.0.0.1", pseudo_labels_port))
        self.pl_socket.setblocking(False)

        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.robot_sock = (robot_ip, robot_port)

        self.accelerator = accelerator
        self.model = utils.get_model(self.device, emg_shape, num_gestures, True)
        self.model.to(self.accelerator)
        self.model.eval()
        self.optimizer = self.model.configure_optimizers()

        self.imu_window_size = 10 if self.device == "myo" else 20  # 200 ms

        self.emg_shape = emg_shape
        self.emg_buffer_size = emg_buffer_size
        self.emg_window_size = emg_window_size_ms * emg_fs // 1000
        self.last_emg_sample = np.zeros((1, *self.emg_shape))

        self.emg_buffer = np.zeros((0, 1, *self.emg_shape))
        self.labels = np.zeros((0,))

        self.voter = majority_vote.MajorityVote(emg_maj_vote_ms * emg_fs // 1000)
        self.gesture_dict = utils.map_class_to_gestures(g.TRAIN_DATA_DIR)

        # Everything else is handled by EMG
        self.arm_movement_list = s.ArmControl._member_names_
        self.arm_calib_data = np.zeros((len(self.arm_movement_list), 4))
        self.arm_calib_deadzones = [0] * len(self.arm_movement_list)
        if not calibrate_imu:
            try:
                self.load_calibration_data()
            except Exception:
                calibrate_imu = True

        if calibrate_imu:
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

        if self.device == "bio":
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
            f"data/{self.device}/imu_calib_data.npz",
            arm_calib_data=self.arm_calib_data,
            arm_calib_deadzones=np.array(self.arm_calib_deadzones),
        )

    def load_calibration_data(self):
        data = np.load(f"data/{self.device}/imu_calib_data.npz")
        self.arm_calib_data = data["arm_calib_data"]
        self.arm_calib_deadzones = list(data["arm_calib_deadzones"])
        log.info(f"Loaded calibration data:\n {self.arm_calib_data}")
        log.info(f"Loaded calibration deadzones: {self.arm_calib_deadzones}")

    def run(self):
        """
        IMU determines the arm movement.
        """

        # TODO intensity of the movement etc
        last_labels = None
        last_gripper_cmd = None
        last_wrist_cmd = None
        last_arm_cmd = None
        while True:
            # t0 = time.perf_counter()

            quats = self.get_imu_data("quat", self.imu_window_size)
            imu_command = self.get_arm_movement(quats)

            emg_data = self.get_emg_data(True, self.emg_window_size)
            preds = self.predict_from_emg(emg_data)
            gripper_cmd, wrist_cmd = self.get_gripper_command(preds)

            labels = self.get_live_labels()
            if labels is not None:
                last_labels = labels

            # only do Wrist/Gripper actions when arm is neutral, force neutral
            # only do Gripper actions when wrist is neutral
            if last_arm_cmd != s.ArmControl.NEUTRAL:
                gripper_cmd = s.GripperControl.NEUTRAL
                wrist_cmd = s.WristControl.NEUTRAL
            elif last_wrist_cmd != s.WristControl.NEUTRAL:
                gripper_cmd = s.GripperControl.NEUTRAL

            # Live labelling stuff
            if emg_data is not None and last_labels is not None:
                self.emg_buffer = np.vstack(
                    (self.emg_buffer, np.reshape(emg_data, (-1, 1, *self.emg_shape)))
                )
                self.labels = np.hstack(
                    (self.labels, np.repeat(last_labels, len(emg_data)))
                )

            # Label buffer filled, do finetuning pass
            if len(self.labels) >= self.emg_buffer_size:
                self.finetune_model(self.emg_buffer, self.labels)
                last_labels = None
                self.emg_buffer = np.zeros((0, 1, *self.emg_shape))
                self.labels = np.zeros((0,))

            # Send commands for arm, wrist, gripper
            if imu_command is not None and imu_command != last_arm_cmd:
                log.info(f"Arm: {imu_command.name}")
                last_arm_cmd = imu_command
                self.send_robot_command(s.to_dict(imu_command))

            if wrist_cmd is not None and wrist_cmd != last_wrist_cmd:
                log.info(f"Wrist: {wrist_cmd.name}")
                last_wrist_cmd = wrist_cmd
                self.send_robot_command(s.to_dict(wrist_cmd))

            if gripper_cmd is not None and gripper_cmd != last_gripper_cmd:
                log.info(f"Gripper: {gripper_cmd.name}")
                last_gripper_cmd = gripper_cmd
                self.send_robot_command(s.to_dict(gripper_cmd))

            # if emg_data is not None:
            #     print("*" * 80)
            #     print("EMG length:", len(emg_data))
            #     print(f"Time taken {time.perf_counter() - t0:.4f} s")

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
        if self.device == "bio":
            # myo is (quat, acc, gyro)
            # bio is (acc, quat)
            odata = np.roll(odata, 4, axis=1)
        # log.info(f"IMU data shape: {odata.shape}")
        vals = odata[:, start:end]
        if channel == "quat":
            # quats' norm is always 1
            vals = vals / np.linalg.norm(vals, axis=1, keepdims=True)
        return vals

    def get_pyr_from_imu(self, quats: np.ndarray):
        """
        Get pitch-yaw-roll from quaternions.

        If multiple quaternions are given, the mean is taken.

        Returns a triplet of pitch, yaw, roll in rads
        """
        if len(quats) == 0:
            return None, None, None

        quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)  # norm per row
        pitch, yaw, roll = get_attitude_from_quats(*quats.T)  # (n, 4) to (4, n)
        pitch, yaw, roll = np.mean(pitch), np.mean(yaw), np.mean(roll)

        log.info(
            f"Pitch: {pitch.item():.3f}, Yaw: {yaw.item():.3f}, Roll: {roll.item():.3f}"
        )
        return pitch, yaw, roll

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
        id = np.argmax(sim_score, axis=1).item(0)
        distance = Quaternion.absolute_distance(
            Quaternion(quats.T), Quaternion(self.arm_calib_data[0].T)
        )
        # log.info(f"Similarity matrix:\n {sim_score}")
        # log.info(f"Distance: {distance}")
        if distance < self.arm_calib_deadzones[id]:
            # Inside of dead zone so assume neutral
            return s.ArmControl[self.arm_movement_list[id]]
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
            data = utils.process_data(data)
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
        samples = torch.from_numpy(data).to(g.ACCELERATOR)
        pred = self.model(samples).cpu().detach().numpy()
        pred = softmax(pred, axis=1)
        # log.info(f"Predictions:\n {np.mean(pred, axis=0)}")
        pred[pred < 0.7] = 0
        return np.argmax(pred, axis=1)

    def get_gripper_command(self, preds: np.ndarray):
        """
        Get the gripper and wrist commands from the predictions.

        Params:
            - preds: np.ndarray, shape (N,). The predictions from the model.
        """
        if preds is None:
            return None, None
        gripper_cmd = s.GripperControl.NEUTRAL
        wrist_cmd = s.WristControl.NEUTRAL
        self.voter.extend(preds)
        maj_vote = self.voter.vote().item(0)
        gesture_str = self.gesture_dict[maj_vote]
        if gesture_str == "Hand_Open":
            gripper_cmd = s.GripperControl.OPEN
        elif gesture_str == "Hand_Close":
            gripper_cmd = s.GripperControl.CLOSE
        elif gesture_str == "Wrist_Flexion":
            wrist_cmd = s.WristControl.FLEXION
        elif gesture_str == "Wrist_Extension":
            wrist_cmd = s.WristControl.EXTENSION
        elif gesture_str == "Thumbs_Down":
            wrist_cmd = s.WristControl.ABDUCTION
        elif gesture_str == "Thumbs_Up":
            wrist_cmd = s.WristControl.ADDUCTION
        elif gesture_str == "Wrist_Pronation":
            cmd = s.ArmControl.PRONATION
        elif gesture_str == "Wrist_Supination":
            cmd = s.ArmControl.SUPINATION

        # log.info(f"{gesture_str} ({maj_vote})")

        return gripper_cmd, wrist_cmd

    def get_live_labels(self):
        """
        Run the socket and return the data.
        """
        try:
            sockdata = np.frombuffer(self.pl_socket.recv(2048), dtype=np.uint8)
            return sockdata
        except Exception:
            return None

    def send_robot_command(self, cmd: dict):
        """
        Send the robot command to the robot.
        """
        self.robot_socket.sendto(json.dumps(cmd).encode(), self.robot_sock)

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
        print(labels.shape)
        self.model.train()
        losses = []
        for _ in range(epochs):
            loss = self.model.training_step((data, labels), 0)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            losses.append(loss.item())
        self.model = self.model
        return losses

    def visualize_imu(self):
        """
        TODO axes are not correct
        https://toptechboy.com/9-axis-imu-lesson-20-vpython-visualization-of-roll-pitch-and-yaw/
        """
        vp.scene.range = 5
        vp.scene.forward = vp.vector(-1, -1, -1)
        vp.scene.width = 600
        vp.scene.height = 600

        frontArrow = vp.arrow(
            length=4, shaftwidth=0.1, color=vp.color.purple, axis=vp.vector(1, 0, 0)
        )
        upArrow = vp.arrow(
            length=6, shaftwidth=0.1, color=vp.color.magenta, axis=vp.vector(0, 1, 0)
        )
        sideArrow = vp.arrow(
            length=2, shaftwidth=0.1, color=vp.color.orange, axis=vp.vector(0, 0, 1)
        )

        pod = vp.box(
            length=6,
            width=3,
            height=1,
            opacity=1,
            pos=vp.vector(
                0,
                0,
                0,
            ),
        )
        while True:
            quats = self.get_imu_data("quats")
            if len(quats) == 0:
                continue
            pit, ya, ro = self.get_pyr_from_imu(quats)
            pit, ya, ro = pit.item(), ya.item(), ro.item()
            vp.rate(50)
            k = vp.vector(
                vp.cos(ya) * vp.cos(pit), vp.sin(pit), vp.sin(ya) * vp.cos(pit)
            )
            y = vp.vector(0, 1, 0)
            s = vp.cross(k, y)
            v = vp.cross(s, k)
            vrot = v * vp.cos(ro) + vp.cross(k, v) * vp.sin(ro)

            frontArrow.axis = k
            sideArrow.axis = vp.cross(k, vrot)
            upArrow.axis = vrot
            pod.axis = k
            pod.up = vrot
            sideArrow.length = 2
            frontArrow.length = 4
            upArrow.length = 1

    def visualize_emg(self):
        self.odh.visualize()


def plot_quaternions_live(odh: OnlineDataHandler):
    # TODO find a better way to watch for new data
    # quat, acc, gyro = 10 dimensions
    global sample_buf, last_shape
    sample_buf = np.zeros((250, 4))
    x_axis = np.arange(len(sample_buf))

    last_shape = (0, 10)

    def update(frame):
        global last_shape, sample_buf

        odata = odh.get_imu_data()
        if odata.shape == last_shape or len(odata) == 0:
            return
        n_new = len(odata) - last_shape[0]
        last_shape = odata.shape
        new_data = odata[-n_new:]  # new data

        quats = new_data[0, :4] / 16000.0
        sample_buf = np.roll(sample_buf, -n_new, axis=0)
        sample_buf[-n_new:] = quats

        # https://stackoverflow.com/questions/54214698/quaternion-to-yaw-pitch-roll#:~:text=Having%20given%20a%20Quaternion%20q,*qy%20%2D%20qz*qz)%3B
        pitch, yaw, roll = get_attitude_from_quats(*quats)

        print("New data len:", len(new_data))
        print(f"Yaw: {yaw:.3f}, Pitch: {pitch:.3f}, Roll: {roll:.3f}")

        plt.cla()

        plt.plot(x_axis, sample_buf[:, 0])
        plt.plot(x_axis, sample_buf[:, 1])
        plt.plot(x_axis, sample_buf[:, 2])
        plt.plot(x_axis, sample_buf[:, 3])

        plt.legend(["w", "x", "y", "z"], loc="upper left")
        plt.ylim(-1.05, 1.05)
        plt.xlim(0, len(sample_buf))
        plt.grid(True, "both", "both")

    plt.figure()
    FuncAnimation(plt.gcf(), update, interval=30)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    from emager_py.utils import set_logging
    import time

    """
    Myo: USB port towards the user: Pitch inverted, Yaw and roll swapped
    """
    set_logging()

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
        g.ROBOT_IP,
        g.ROBOT_PORT,
        # False,
    )
    # odw.visualize_emg()
    odw.run()
    while True:
        t0 = time.perf_counter()
        data = odw.get_imu_data("quat")
        if len(data) == 0:
            continue
        mvmt = odw.get_arm_movement(data)
        print(mvmt)
        print("*" * 80)
        print(f"Time taken {time.perf_counter() - t0:.4f} s")
        time.sleep(0.1)
