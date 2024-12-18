import os


class NfcPaths:
    """
    Opinionated helper to set up paths for an EMG experiment.

    General structure goes:

    'base/trial_name/{subfolders go here}'

    Base usually is something like `data/` or `data/subject_id/`
    """

    base: str
    trial: int

    gestures: str = "gestures/"

    # Folders
    train: str = "train/"
    test: str = "test/"
    fine: str = "fine/"
    models: str = "models/"
    memory: str = "memories/"

    # Files
    model: str = "model.pth"
    live_data: str = "live_"
    results: str = "results.csv"

    def __init__(self, base="data", trial: str | None = None):
        """
        Initialize paths with a base directory and trial number.
        """
        self.set_base(base)
        if trial is None:
            trial = self.get_next_trial()
        self.set_trial(trial)

    def set_base(self, base: str):
        self.base = base
        _set_dir(base)

        return self

    def set_trial(self, trial):
        if not isinstance(trial, str):
            trial = str(trial)
        self.trial = trial
        _set_dir(self.get_experiment_dir())
        return self

    def get_experiment_dir(self):
        return f"{self.base}/{self.trial}/"

    def set_model(self, name: str):
        """
        Set model name. Does not require .pth extension.
        """
        if not name.endswith(".pth"):
            name += ".pth"
        self.model = name
        return self

    def get_train(self):
        return f"{self.get_experiment_dir()}{self.train}"

    def get_test(self):
        return f"{self.get_experiment_dir()}{self.test}"

    def get_fine(self):
        return f"{self.get_experiment_dir()}{self.train}"

    def get_models(self):
        models = f"{self.get_experiment_dir()}{self.models}"
        _set_dir(models)
        return models

    def get_gestures(self):
        return f"{self.get_experiment_dir()}{self.gestures}"

    def get_memory(self):
        mem = f"{self.get_experiment_dir()}{self.memory}"
        _set_dir(mem)
        return mem

    # ------- Files -------

    def get_model(self):
        return f"{self.get_experiment_dir()}{self.model}"

    def get_results(self):
        return f"{self.get_experiment_dir()}{self.results}"

    def get_live(self):
        return f"{self.get_experiment_dir()}{self.live_data}"

    def get_next_trial(self):
        try:
            trials = os.listdir(self.base)
            if not trials:
                return 0
            trials = list(filter(lambda x: x.isnumeric(), trials))
            return max([int(t) for t in trials]) + 1
        except FileNotFoundError:
            return 0


def _set_dir(dir: str):
    if not dir.endswith("/"):
        dir += "/"

    if not os.path.exists(dir):
        os.makedirs(dir, exist_ok=True)

    return dir
