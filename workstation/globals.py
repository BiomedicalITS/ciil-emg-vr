try:
    import torch

    ACCELERATOR = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
except ImportError:
    pass

DEVICE = "myo"  # "emager", "myo" or "bio"

BASE_DIR = f"data/{DEVICE}/"
TRAIN_DATA_DIR = BASE_DIR + "train/"
FINETUNE_DATA_DIR = BASE_DIR + "finetune/"
MODEL_PATH = BASE_DIR + "model.pth"

# LIBEMG_GESTURE_IDS = [1, 2, 3, 4, 5] # minimal
# LIBEMG_GESTURE_IDS = [1, 2, 3, 4, 5, 6, 7]  # + supi + prona
LIBEMG_GESTURE_IDS = [1, 2, 3, 4, 5, 14, 15]  # + abduction + adduction
# LIBEMG_GESTURE_IDS = [1, 2, 3, 4, 5, 6, 7, 14, 15]  # + supi + prona + abduction + adduction

PEUDO_LABELS_PORT = 5111
ROBOT_PORT = 5112
ROBOT_IP = "192.168.50.39"  # nvidia

EMG_DATA_SHAPE = (4, 16) if DEVICE == "emager" else (1, 8)
EMG_SAMPLING_RATE = 1000 if DEVICE == "emager" else 1500 if DEVICE == "bio" else 200
EMG_SCALING_FACTOR = (
    2.0**15 if DEVICE == "emager" else 0.008 if DEVICE == "bio" else 128.0
)
EMG_NOTCH_FREQ = 50
EMG_WINDOW_MS = 25
EMG_MAJ_VOTE_MS = 200
EMG_RUNNING_MEAN_LEN = EMG_WINDOW_MS * EMG_SAMPLING_RATE // 1000
