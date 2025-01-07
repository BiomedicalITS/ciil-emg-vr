import time
import numpy as np
import matplotlib.pyplot as plt

import sifi_bridge_py as sbp

# GESTURES = ["Hand_Close", "Chuck_Grip", "No_Motion", "Index_Pinch"]
GESTURES = ["Hand_Close"]
DATA_PATH = "paper/data/emg_data_example_%s.csv"
FIG_PATH = "paper/figures/emg_data_example_%s.png"


def sample():
    sb = sbp.SifiBridge()
    sb.connect("BioArmband")
    sb.set_memory_mode("streaming")
    sb.set_channels(emg=True)
    sb.set_filters(True)
    sb.configure_emg(notch_freq=60)

    channels = [f"emg{i}" for i in range(8)]

    for gesture in GESTURES:
        data = [[] for _ in range(8)]
        input(f"Do {gesture} and press enter.")

        # Warmup for 0.5s for EMG to be unsaturated
        sb.start()
        time.sleep(1)
        t0 = time.time()
        while True:
            sb.get_emg()
            if time.time() - t0 > 0.5:
                break

        # Start acquisition
        t0 = time.time()
        while time.time() - t0 < 2:
            packet = sb.get_emg()
            for i, ch in enumerate(channels):
                data[i].extend(packet["data"][ch])
        sb.stop()

        data = np.array(data)
        t = np.array([i / 2000 for i in range(len(data[0]))])
        np.savetxt((DATA_PATH % gesture), np.vstack([t, data]), delimiter=",")

    sb.stop()


def plot_heatmap():
    # How to do a heatmap: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html
    for gesture in GESTURES:
        # (9, n_samples)
        data = np.loadtxt(DATA_PATH % gesture, delimiter=",")
        data = data[1:] * 1e6  # mV
        # (8, n_samples)
        data = np.mean(np.abs(data), keepdims=True, axis=1)
        # reordering
        data = data[[4, 6, 3, 0, 7, 1, 2, 5]]

        print(np.max(data))
        channels = list(range(8))

        # Plot and format
        fig, ax = plt.subplots()
        im = ax.imshow(data.T, vmin=0, vmax=40)
        ax.set_xticks(np.arange(8), labels=channels)
        ax.set_xlabel("Channel")
        ax.set_yticks([])
        fig.tight_layout()

        kw = dict(horizontalalignment="center", verticalalignment="center")
        texts = []
        textcolors = ("white", "black")
        for i in range(len(data)):
            v = data[i, 0]
            # medpoint = (np.max(data) + np.min(data)) / 2
            # th = medpoint + ((np.max(data) - medpoint) / 2)
            th = 20
            kw.update(color=textcolors[int(v > th)])
            text = im.axes.text(i, 0, f"{v:.2f}", **kw)
            texts.append(text)

        # Create colorbar
        cbar = ax.figure.colorbar(im, ax=ax)
        cbar.ax.set_ylabel("EMG (uV)", rotation=-90, va="bottom")

        plt.savefig(FIG_PATH % gesture, bbox_inches="tight", dpi=400)


def plot_data():
    for gesture in GESTURES:
        # (9, n_samples)
        data = np.loadtxt(DATA_PATH % gesture, delimiter=",")
        t = data[0]
        data = data[1:] * 1e6  # mV
        # (8, n_samples)
        # data = np.mean(np.abs(data), keepdims=True, axis=1)
        # reordering
        data = data[[4, 6, 3, 0, 7, 1, 2, 5]]

        print(np.max(data))

        # Plot and format
        fig, ax = plt.subplots()
        plt.plot(t, data.T)
        plt.show()
        # plt.savefig(FIG_PATH % gesture, bbox_inches="tight", dpi=400)


if __name__ == "__main__":
    # sample()
    plot_heatmap()
    # plot_data()
