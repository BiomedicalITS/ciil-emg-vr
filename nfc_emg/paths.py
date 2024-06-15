from dataclasses import dataclass


@dataclass
class NfcPaths:
    base: str = "data/"
    gestures: str = "gestures/"
    model: str = "model.pth"
    train: str = "train/"
    fine: str = "fine/"
    test: str = "test/"
    imu_calib: str = "imu_calib_data.npz"

    def __init__(self, base="data/"):
        self.set_base_dir(base)

    def set_base_dir(self, base_dir: str):
        if not base_dir.endswith("/"):
            base_dir += "/"
        self.base = base_dir
        self.gestures = self.base + "gestures/"
        self.model = self.base + "model.pth"
        self.train = self.base + "train/"
        self.fine = self.base + "fine/"
        self.test = self.base + "test/"
        self.imu_calib = self.base + "imu_calib_data.npz"

    def set_model_name(self, name: str):
        """
        Set model name. Does not require .pth extension.
        """
        if name.endswith(".pth"):
            name = name[:-4]
        self.model = self.base + name + ".pth"
