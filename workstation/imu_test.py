import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from libemg.data_handler import OnlineDataHandler

import utils
import globals as g


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
        qw, qx, qy, qz = quats
        yaw = np.arctan2(
            2.0 * (qy * qz + qw * qx), qw * qw - qx * qx - qy * qy + qz * qz
        )
        pitch = np.arcsin(-2.0 * (qx * qz - qw * qy))
        roll = np.arctan2(
            2.0 * (qx * qy + qw * qz), qw * qw + qx * qx - qy * qy - qz * qz
        )

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
    # Setup the streamers
    utils.setup_streamer()

    # Create data handler and model
    odh = utils.get_online_data_handler(
        g.EMG_SAMPLING_RATE,
        notch_freq=g.EMG_NOTCH_FREQ,
        use_imu=True,
        max_buffer=1,
    )
    # And now main loop
    main_loop(odh)
