import vpython as vp
import logging as log
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import socket

from emager_py import majority_vote

import numpy as np
import torch

from libemg.data_handler import OnlineDataHandler

import utils
import globals as g


def get_attitude_from_quats(qw, qx, qy, qz):
    yaw = np.arctan2(2.0 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz)

    aasin = qx * qz - qw * qy
    # if aasin > 0.97:
    #    pitch = np.pi / 2.0
    # elif aasin < -0.97:
    #    pitch = -np.pi / 2.0
    if False:
        pass
    else:
        pitch = np.arcsin(-2.0 * aasin)

    roll = np.arctan2(2.0 * (qx * qy + qw * qz), qw * qw + qx * qx - qy * qy - qz * qz)
    return pitch, yaw, roll


class OnlineDataWrapper:
    def __init__(self, emg_shape: tuple, emg_buffer_size: int, udp_port: int = 5111):
        """
        Main object to do the NFC-EMG stuff.

        Starts listening to the data stream.
        """
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.socket.bind(("127.0.0.1", udp_port))
        self.socket.setblocking(False)

        self.odh = OnlineDataHandler(
            emg_arr=True, imu_arr=True, max_buffer=emg_buffer_size
        )
        self.model = utils.get_model(False, len(g.LIBEMG_GESTURE_IDS))
        self.model.eval()
        self.optimizer = self.model.configure_optimizers()

        self.emg_buffer_size = emg_buffer_size
        self.emg_shape = emg_shape
        self.last_emg_sample = np.zeros((1, *self.emg_shape))

        utils.setup_streamer()
        self.start_listening()

    def start_listening(self):
        self.odh.start_listening()

    def stop_listening(self):
        self.odh.stop_listening()

    def run(self):
        voter = majority_vote.MajorityVote(
            g.EMG_MAJ_VOTE_MS * g.EMG_SAMPLING_RATE // 1000
        )
        gesture_dict = utils.map_class_to_gestures(g.TRAIN_DATA_DIR)

        while True:
            pitch, yaw, roll = self.run_imu()
            emg_data = self.run_emg()
            labels = self.run_socket()
            preds = self.run_model(emg_data)
            if preds.size > 0:
                voter.extend(preds)
                maj_vote = voter.vote().item(0)
                log.info(f"{gesture_dict[maj_vote]} ({maj_vote})")

    def run_imu(self):
        """
        Get IMU quaternions, return the attitude readings. Clear the odh buffer.
        """
        odata = self.odh.get_imu_data()
        if len(odata) == 0:
            return None, None, None
        # log.info(f"IMU data len: {len(odata)}")
        quats: np.ndarray = odata[:, :4] / np.linalg.norm(
            odata[:, :4], axis=1, keepdims=True
        )
        pitch, yaw, roll = get_attitude_from_quats(*quats.T)
        self.odh.raw_data.reset_imu()
        return pitch, yaw, roll

    def run_emg(self):
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
        pdata = utils.process_data(data)
        return pdata[pos + 1 :]

    def run_model(self, data: np.ndarray) -> np.ndarray:
        """
        Run the model on the given data.
        """
        if data.size == 0:
            return np.zeros((0,), np.uint8)
        data = data.reshape(-1, 1, *self.emg_shape)
        samples = torch.from_numpy(data).to(g.ACCELERATOR)
        pred = self.model(samples)
        pred = pred.cpu().detach().numpy()
        return np.argmax(pred, axis=1)

    def run_socket(self):
        """
        Run the socket and return the data.
        """
        try:
            sockdata = np.frombuffer(self.socket.recv(2048), dtype=np.uint8)
            return sockdata
        except Exception:
            return None

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
            pitch, yaw, roll = self.run_imu()
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


def main_loop(odh: OnlineDataHandler):
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
    ani = FuncAnimation(plt.gcf(), update, interval=30)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    import time
    from emager_py.utils import set_logging

    set_logging()

    odw = OnlineDataWrapper(g.EMG_DATA_SHAPE, g.EMG_SAMPLING_RATE)
    odw.run()
    while True:
        print("*" * 80)
        t0 = time.perf_counter()
        data = odw.run_emg()
        print("New data shape", data.shape)
        preds = odw.run_model(data)
        print(f"Time taken {time.perf_counter() - t0:.4f} s")
        print(preds)
        # odw.visualize_emg()
