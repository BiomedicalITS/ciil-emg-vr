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
    for i in os.listdir(lightning_logs_path):
        if "version" in i:
            num = int(i.split("_")[1])
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


def get_model():
    """
    Load the most recent model checkpoint and return it
    """

    model = EmgCNN.load_from_checkpoint(
        checkpoint_path=get_most_recent_checkpoint(),
    )
    model = TuningEmgCNN(model, num_classes=len(g.LIBEMG_GESTURE_IDS)).to(g.ACCELERATOR)
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
    # standard preprocessing pipeline
    data = np.abs(data)

    # finish by normalizing samples
    orig_shape = data.shape
    if data.ndim != 2:
        data = data.reshape(data.shape[0], -1)

    normalized = normalize(data, axis=1)
    if orig_shape != normalized.shape:
        normalized = normalized.reshape(orig_shape)
    return normalized.astype(np.float32)


def visualize():
    p = setup_streamer()
    odh = get_online_data_handler(
        g.EMG_SAMPLING_RATE, notch_freq=g.EMG_NOTCH_FREQ, use_imu=g.USE_IMU
    )
    odh.visualize_channels(list(range(8)), 3 * g.EMG_SAMPLING_RATE)
    return p


if __name__ == "__main__":
    p = setup_streamer()
    input("Press Enter to quit")
    p = visualize()

    p = get_most_recent_checkpoint()
    print(p)
