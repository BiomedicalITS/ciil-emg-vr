from libemg.feature_extractor import FeatureExtractor
from nfc_emg.sensors import EmgSensorType

try:
    import torch
    from lightning.pytorch import seed_everything

    __SEED = 310
    seed_everything(__SEED)
    ACCELERATOR = (
        "cuda"
        if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available() else "cpu"
    )
    if ACCELERATOR == "cuda":
        torch.set_float32_matmul_precision("high")

except ImportError:
    pass

# https://github.com/libemg/LibEMGGestures
BASIC_SET = [1, 2, 3, 4, 5, 26, 30]
FUNCTIONAL_SET = [1, 2, 3, 4, 5, 8, 26, 30]  # w/o wrist up/down
# FUNCTIONAL_SET = [1, 2, 3, 4, 5, 8, 17, 18, 26, 30]  # w/ wrist up/down
FINE_SET = [1, 2, 3, 8, 14, 26, 30]

_fe = FeatureExtractor()
FEATURES = _fe.get_feature_groups()["TDPSD"]
# FEATURES = ["MAV"]
SUBJECT = "vr"
SENSOR = EmgSensorType.BioArmband

WAIT_TCP_PORT = 12346
PSEUDO_LABELS_PORT = 12350
PREDS_PORT = 12347
PREDS_IP = "127.0.0.1"
