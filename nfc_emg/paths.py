from dataclasses import dataclass
import os


@dataclass
class NfcPaths:
    base: str = "data/"
    gestures: str = "gestures/"
    model: str = "model.pth"
    train: str = "train/"
    fine: str = "fine/"
    test: str = "test/"
    results: str = "results/"
    imu_calib: str = "imu_calib_data.npz"

    def __init__(self, base="data/", trial_number: int | None = None):
        self.base = base
        if trial_number is None:
            trial_number = self.get_next_trial()
        self.set_trial_number(trial_number)
        self.set_base_dir(base)

    def set_base_dir(self, base_dir: str):
        if not base_dir.endswith("/"):
            base_dir += "/"

        if not os.path.exists(self.base):
            os.makedirs(self.base, exist_ok=True)

        self.base = base_dir
        self.set_trial_number(self.trial_number)

    def set_trial_number(self, number):
        self.trial_number = number
        if not os.path.exists(self.base + str(self.trial_number)):
            os.makedirs(self.base, exist_ok=True)

        self.gestures = f"{self.base}/{self.trial_number}/gestures/"
        self.train = f"{self.base}/{self.trial_number}/train/"
        self.fine = f"{self.base}/{self.trial_number}/fine/"
        self.test = f"{self.base}/{self.trial_number}/test/"
        self.results = f"{self.base}/{self.trial_number}/results.json"
        self.imu_calib = f"{self.base}/{self.trial_number}/imu_calib_data.npz"
        self.model = self.set_model_name(self.model.split("/")[-1])

    def set_model_name(self, name: str):
        """
        Set model name. Does not require .pth extension.
        """
        if name.endswith(".pth"):
            name = name[:-4]
        self.model = f"{self.base}/{self.trial_number}/{name}.pth"
        return self.model

    def get_next_trial(self):
        try:
            trials = os.listdir(self.base)
            if not trials:
                return 0
            trials = list(filter(lambda x: x.isnumeric(), trials))
            return max([int(t) for t in trials]) + 1
        except FileNotFoundError:
            return 0
