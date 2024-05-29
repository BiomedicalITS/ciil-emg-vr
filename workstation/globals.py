try:
    import torch

    ACCELERATOR = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
except ImportError:
    pass

DEVICE = "myo"  # "emager", "myo" or "bioarmband"

TRAIN_DATA_DIR = f"data/{DEVICE}/train/"
FINETUNE_DATA_DIR = f"data/{DEVICE}/finetune/"
LIBEMG_GESTURE_IDS = [1, 2, 3, 4, 5]  # pinch = 26 ?

PEUDO_LABELS_PORT = 5111
ROBOT_PORT = 5112
ROBOT_IP = "192.168.50.39"

EMG_DATA_SHAPE = (4, 16) if DEVICE == "emager" else (1, 8)
EMG_SAMPLING_RATE = (
    1000 if DEVICE == "emager" else 1500 if DEVICE == "bioarmband" else 200
)
EMG_NOTCH_FREQ = 50
EMG_WINDOW_MS = 25
EMG_MAJ_VOTE_MS = 200
EMG_RUNNING_MEAN_LEN = EMG_WINDOW_MS * EMG_SAMPLING_RATE // 1000
