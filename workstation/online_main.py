import vpython as vp
import logging as log
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import socket
import json

from emager_py import majority_vote

import numpy as np
import torch

from libemg.data_handler import OnlineDataHandler

import utils
import globals as g


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
        emg_shape: tuple,
        emg_buffer_size: int,
        emg_fs: int,
        emg_maj_vote_ms: int,
        num_gestures: int,
        pseudo_labels_port: int = 5111,
        robot_ip: str = "nvidia",
        robot_port: int = 5112,
    ):
        """
        Main object to do the NFC-EMG stuff.

        Starts listening to the data stream.
        """
        self.device = device

        self.pl_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.pl_socket.bind(("127.0.0.1", pseudo_labels_port))
        self.pl_socket.setblocking(False)

        self.robot_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.robot_sock = (robot_ip, robot_port)

        self.odh = OnlineDataHandler(
            emg_arr=True, imu_arr=True, max_buffer=emg_buffer_size
        )
        self.model = utils.get_model(True, num_gestures)
        self.model.eval()
        self.optimizer = self.model.configure_optimizers()

        self.emg_buffer_size = emg_buffer_size
        self.emg_shape = emg_shape
        self.last_emg_sample = np.zeros((1, *self.emg_shape))

        self.voter = majority_vote.MajorityVote(emg_maj_vote_ms * emg_fs // 1000)
        self.gesture_dict = utils.map_class_to_gestures(g.TRAIN_DATA_DIR)

        utils.setup_streamer(device)
        self.start_listening()

    def start_listening(self):
        self.odh.start_listening()

    def stop_listening(self):
        self.odh.stop_listening()

    def run(self):
        # TODO intensity of the movement etc
        last_arm_movement = {"arm": "none"}
        last_gripper_cmd = {"gripper": "none"}
        while True:
            quats = self.get_imu_data("quat")
            pyr = self.get_pyr_from_imu(quats)
            arm_movement = self.get_arm_movement(*pyr)

            emg_data = self.get_emg()
            preds = self.predict_from_emg(emg_data)
            gripper_cmd = self.get_gripper_command(preds)

            # labels = self.get_live_labels()
            if arm_movement["arm"] != "none" and arm_movement != last_arm_movement:
                log.info(f"Arm movement: {arm_movement}")
                last_arm_movement = arm_movement
                self.send_robot_command(arm_movement)
            if gripper_cmd["gripper"] != "none" and gripper_cmd != last_gripper_cmd:
                log.info(f"Gripper command: {gripper_cmd}")
                last_gripper_cmd = gripper_cmd
                self.send_robot_command(gripper_cmd)

    def get_imu_data(self, channel: str = "quat"):
        """
        Get IMU quaternions, return the attitude readings. Clear the odh buffer.

        Params:
            - channel: str, "quat", "acc" or "both"
        """
        start = 4 if channel == "acc" else 0
        end = 4 if channel == "quat" else 7
        odata = self.odh.get_imu_data()
        if len(odata) == 0:
            return np.zeros((0, end - start))
        # log.info(f"IMU data len: {len(odata)}")
        vals: np.ndarray = odata[:, start:end]
        self.odh.raw_data.reset_imu()
        return vals

    def get_pyr_from_imu(self, quats: np.ndarray):
        """
        Get pitch-yaw-roll from quaternions.

        If multiple quaternions are given, the mean is taken.

        Returns a triplet of pitch, yaw, roll in rads
        """
        if len(quats) == 0:
            return None, None, None
        elif len(quats) > 1:
            quats = np.mean(quats, axis=0)

        quats = quats / np.linalg.norm(quats)
        pitch, yaw, roll = get_attitude_from_quats(*quats.T)
        log.info(
            f"Pitch: {pitch.item():.3f}, Yaw: {yaw.item():.3f}, Roll: {roll.item():.3f}"
        )
        return pitch.item(), yaw.item(), roll.item()

    def get_arm_movement(self, pitch, yaw, roll, dead_zone=0.5):
        """
        Get the movement from the pitch-yaw-roll values.

        Pitch = [-pi/2, pi/2]
        Yaw = [-pi, pi]
        Roll = [-pi, pi]
        """
        ret = {"arm": "none"}
        if pitch is None:
            return ret
        pitch, yaw, roll = 2 * pitch / np.pi, yaw / np.pi, roll / np.pi  # [-1, 1]
        triplet = np.array([pitch, yaw, roll])
        biggest = np.argmax(np.abs(triplet))

        if np.abs(triplet[biggest]) < dead_zone:
            return ret
        elif biggest == 0:
            # pitch
            if triplet[biggest] > 0:
                ret["arm"] = "backward"
            else:
                ret["arm"] = "forward"
        elif biggest == 1:
            # yaw
            if triplet[biggest] > 0:
                ret["arm"] = "right"
            else:
                ret["arm"] = "left"
        else:
            # roll
            if triplet[biggest] > 0:
                ret["arm"] = "up"
            else:
                ret["arm"] = "down"
        return ret

    def get_emg(self, process=True):
        odata = self.odh.get_data()
        if len(odata) < self.emg_buffer_size:
            return np.zeros((0, *self.emg_shape), np.float32)
        pos = np.argwhere((self.last_emg_sample == odata).all(axis=1))
        if pos.size == 0:
            # if empty, means we must take the entire buffer
            pos = np.array([0])
        elif pos.size > 1:
            log.error("Multiple positions found: ", pos)
        pos = pos.item(0)
        if pos == len(odata) - 1:
            # No new values
            return np.zeros((0, *self.emg_shape), np.float32)
        self.last_emg_sample = odata[-1]
        data = odata.reshape(-1, *self.emg_shape)
        if process:
            data = utils.process_data(data, self.device)
        return data[pos + 1 :]

    def predict_from_emg(self, data: np.ndarray):
        """
        Run the model on the given data.

        TODO show confidencev%
        """
        if len(data) == 0:
            return np.zeros((0,), np.uint8)
        data = data.reshape(-1, 1, *self.emg_shape)
        samples = torch.from_numpy(data).to(g.ACCELERATOR)
        pred = self.model(samples)
        pred = pred.cpu().detach().numpy()
        return np.argmax(pred, axis=1)

    def get_gripper_command(self, preds: np.ndarray):
        """
        Get the gripper command from the predictions.

        Params:
            - preds: np.ndarray, shape (N,). The predictions from the model.
        """
        cmd = {"gripper": "none"}
        if len(preds) == 0:
            return cmd
        self.voter.extend(preds)
        maj_vote = self.voter.vote().item(0)
        gesture_str = self.gesture_dict[maj_vote]
        if gesture_str == "Hand_Open":
            cmd["gripper"] = "open"
        elif gesture_str == "Hand_Close":
            cmd["gripper"] = "close"
        elif gesture_str == "Wrist_Flexion":
            pass
        elif gesture_str == "Wrist_Extension":
            pass

        log.info(f"{gesture_str} ({maj_vote})")

        return cmd

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
        self.model = self.model.train()
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
            pitch, yaw, roll = self.get_imu_data()
            if pitch is None:
                continue
            pit, ya, ro = pitch[-1], yaw[-1], roll[-1]
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

    """
    Myo: USB port towards the user: Pitch inverted, Yaw and roll swapped
    """
    set_logging()

    odw = OnlineDataWrapper(
        g.DEVICE,
        g.EMG_DATA_SHAPE,
        g.EMG_SAMPLING_RATE,
        g.EMG_SAMPLING_RATE,
        g.EMG_MAJ_VOTE_MS,
        len(g.LIBEMG_GESTURE_IDS),
        g.PEUDO_LABELS_PORT,
        g.ROBOT_IP,
        g.ROBOT_PORT,
    )
    odw.run()
