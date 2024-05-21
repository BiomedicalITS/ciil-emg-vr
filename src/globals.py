import torch

ACCELERATOR = (
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)

USE_MYO = True
USE_IMU = True

OFFLINE_DATA_DIR = "data/offline/"
LIBEMG_GESTURE_IDS = [1, 2, 3, 4, 5]  # pinch = 26 ?

EMG_DATA_SHAPE = (1, 8)
EMG_NOTCH_FREQ = 50
EMG_WINDOW_MS = 25
EMG_MAJ_VOTE_MS = 200
EMG_SAMPLING_RATE = 200 if USE_MYO else 1500
EMG_RUNNING_MEAN_LEN = EMG_WINDOW_MS * EMG_SAMPLING_RATE // 1000
