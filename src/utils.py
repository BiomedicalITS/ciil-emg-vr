import os
import numpy as np
from sklearn.preprocessing import normalize

from libemg.streamers import sifibridge_streamer, myo_streamer
from libemg.data_handler import OnlineDataHandler
from libemg.filtering import Filter


from models import EmgCNN, TuningEmgCNN
import globals as g


def get_most_recent_checkpoint(lightning_logs_path: str = "./lightning_logs") -> str:
    max = 0
    for folder in os.listdir(lightning_logs_path):
        if "version" not in folder:
            continue

        with open(f"{lightning_logs_path}/{folder}/hparams.yaml", "r") as f:
            text = f.readline()
            if text.startswith("{}"):
                continue

        num = int(folder.split("_")[1])
        if num > max:
            max = num
    ckpt_name = os.listdir(f"{lightning_logs_path}/version_{max}/checkpoints")[0]
    model_ckpt_path = f"{lightning_logs_path}/version_{max}/checkpoints/{ckpt_name}"

    return model_ckpt_path


def get_filter(
    sampling_rate: float,
    bandpass_freqs: float | list = [20, 350],
    notch_freq: int = 50,
):
    if isinstance(bandpass_freqs, float) or isinstance(bandpass_freqs, int):
        bandpass_freqs = [bandpass_freqs]

    # Create some filters
    fi = Filter(sampling_frequency=sampling_rate)
    fi.install_filters({"name": "notch", "cutoff": notch_freq, "bandwidth": 3})
    if len(bandpass_freqs) == 2 and sampling_rate > bandpass_freqs[1] * 2:
        fi.install_filters(
            filter_dictionary={"name": "bandpass", "cutoff": bandpass_freqs, "order": 4}
        )
    else:
        fi.install_filters(
            {"name": "highpass", "cutoff": bandpass_freqs[0], "order": 2}
        )
    return fi


def get_model(finetune: bool = False, num_classes: int = 6):
    """
    Load the most recent model checkpoint and return it
    """
    model_ckpt = get_most_recent_checkpoint()
    print("Loading model from:", model_ckpt)
    model = EmgCNN.load_from_checkpoint(checkpoint_path=model_ckpt)
    if not finetune:
        return model

    model = TuningEmgCNN(model, num_classes=num_classes)
    return model


def setup_streamer():
    # Setup the streamers
    if g.USE_MYO:
        # Create Myo Streamer
        return myo_streamer(filtered=False, imu=g.USE_IMU)
    else:
        return sifibridge_streamer(
            version="1_1",
            emg=True,
            imu=g.USE_IMU,
            notch_on=True,
            notch_freq=g.EMG_NOTCH_FREQ,
        )


def get_online_data_handler(
    sampling_rate: float,
    bandpass_freqs: float | list = [20, 350],
    notch_freq: int = 50,
    use_imu: bool = False,
) -> OnlineDataHandler:
    if not isinstance(bandpass_freqs, list) or isinstance(bandpass_freqs, tuple):
        bandpass_freqs = [bandpass_freqs]

    # Create Online Data Handler - This listens for data
    odh = OnlineDataHandler(imu_arr=use_imu)
    odh.install_filter(get_filter(sampling_rate, bandpass_freqs, notch_freq))
    odh.start_listening()
    return odh


def process_data(data: np.ndarray) -> np.ndarray:
    # Assumes data is prefiltered

    # rectify
    data = np.abs(data)
    # data = tke(data)
    data = moving_average(data, g.EMG_RUNNING_MEAN_LEN)

    # normalize
    if g.USE_MYO:
        return data.astype(np.float32) / 128.0

    return data.astype(np.float32) / (5 * 10e-3)


def moving_average(x: np.ndarray, N: int):
    orig_shape = x.shape
    x = x.reshape(-1, np.prod(x.shape[1:]))

    for c in range(x.shape[1]):
        x[:, c] = np.convolve(x[:, c], np.ones(N) / N, mode="same")
    return x.reshape(orig_shape)


def tke(data):
    data = np.vstack((data[0:1], data))
    data = np.vstack((data, data[-1:]))
    data = data[1:-1] ** 2 - data[:-2] * data[2:]
    return data


def visualize():
    p = setup_streamer()
    odh = get_online_data_handler(
        g.EMG_SAMPLING_RATE, notch_freq=g.EMG_NOTCH_FREQ, use_imu=g.USE_IMU
    )
    odh.visualize_channels(list(range(8)), 3 * g.EMG_SAMPLING_RATE)
    return p


if __name__ == "__main__":
    # p = setup_streamer()
    # p = visualize()

    p = get_most_recent_checkpoint()
    print(p)
