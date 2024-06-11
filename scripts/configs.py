DEVICE = "myo"  # "emager", "myo" or "bio"
# LIBEMG_GESTURE_IDS = [1, 2, 3, 8, 26, 30]
# LIBEMG_GESTURE_IDS = [1, 2, 3, 4, 5, 14, 15]  # + abduction + adduction

try:
    import torch

    ACCELERATOR = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
except ImportError:
    pass

PEUDO_LABELS_PORT = 5111
PREDS_PORT = 5112
PREDS_IP = "127.0.0.1"

LIBEMG_GESTURES_DIR = "data/gestures/"

EMG_DATA_SHAPE = (4, 16) if DEVICE == "emager" else (1, 8)
EMG_SAMPLING_RATE = 1000 if DEVICE == "emager" else 1500 if DEVICE == "bio" else 200

EMG_NOTCH_FREQ = 50
EMG_WINDOW_MS = 25
EMG_MAJ_VOTE_MS = 200
EMG_RUNNING_MEAN_LEN = EMG_WINDOW_MS * EMG_SAMPLING_RATE // 1000
