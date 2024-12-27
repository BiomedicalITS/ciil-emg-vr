import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import logging as log

from sklearn.metrics import accuracy_score

from libemg.feature_extractor import FeatureExtractor


from nfc_emg.sensors import EmgSensorType
from nfc_emg import utils

from experiment.config import Config, ExperimentStage
from experiment.memory import Memory


class SubjectResults:
    def __init__(
        self,
        subject: int,
        adaptation: bool,
        stage=ExperimentStage.SG_PRE_TEST,
        sensor=EmgSensorType.BioArmband,
        features="TDPSD",
    ):
        log.info(f"Loading {subject=}, {adaptation=}, {stage=}, {sensor=}, {features=}")

        self.create_config(subject, sensor, features, stage, adaptation)
        self.set_stage(stage)

    def create_config(self, subject, sensor, features, stage, adaptation):
        self.subject = subject
        self.adaptation = adaptation
        self.sensor = sensor
        self.features = features

        log.info(f"Creating Config object for {subject=}, {stage=}, {adaptation=}")

        self.config = Config(
            subject_id=self.subject,
            sensor_type=self.sensor,
            features=self.features,
            stage=stage,
            adaptation=self.adaptation,
        )
        self.config.paths.gestures = "data/gestures/"
        return self

    def set_stage(self, stage: ExperimentStage):
        """Set the stage the analysis. Sets the `self.stage` and `self.config` attributes.

        Args:
            stage (ExperimentStage): Stage of the experiment.

        Returns:
            self: The current instance.
        """
        self.stage = stage
        log.info(f"Set {stage=} and regenerating Config")
        self.create_config(
            self.subject, self.sensor, self.features, stage, self.adaptation
        )
        return self

    def load_model_eval_metrics(self):
        """Load model evaluation metrics.

        Returns:
            dict: Evaluation metrics.
        """
        with open(
            self.config.paths.get_experiment_dir()
            + f"results_{'pre' if self.stage <= ExperimentStage.GAME else 'post'}.json",
            "r",
        ) as f:
            d = json.load(f)
            if "CONF_MAT" in d.keys():
                d["CONF_MAT"] = np.array(d["CONF_MAT"])
            return d

    def find_memory_ids(self):
        mem_dir = self.config.paths.get_memory()
        files = os.listdir(mem_dir)
        memories = list(filter(lambda x: x.startswith("classifier_memory"), files))
        return sorted([int(mem.split("_")[-1].split(".")[0]) for mem in memories])

    def load_memory(self, mem_id: int):
        """Load a memory file from the subject's memory directory.

        Args:
            mem_id (int): Memory ID. For the latest one, use -1
        """
        if mem_id == -1:
            mems = self.find_memory_ids()
            mem_id = max(mems)

        return Memory().from_file(self.config.paths.get_memory(), mem_id)

    def load_predictions(self):
        """Load predictions from the subject's predictions file.

        Returns:
            tuple: timestamps, predictions, windows
        """
        arr = np.loadtxt(self.config.paths.get_live() + "preds.csv", delimiter=",")
        log.info(f"Predictions file has shape {arr.shape}")
        timestamps = arr[:, 0]
        preds = arr[:, 1]
        data = arr[:, 2:].reshape(
            len(arr),
            np.prod(self.config.sensor.emg_shape),
            -1,
        )

        features = FeatureExtractor().extract_features(
            self.config.features, data, array=True
        )
        log.info(
            f"Loaded {len(preds)} predictions with features of shape {features.shape}"
        )
        return timestamps, preds, features

    def find_unity_logs(self):
        """Find unity logs from `self`, they should be in the root of experiment dir as OL_....txt

        Args:
            dir (str): Directory in which to log for logs

        Returns:
            list[str]: List of log files found.
        """
        base = self.config.paths.get_experiment_dir()
        files = os.listdir(base)
        logs = list(filter(lambda x: x.startswith("OL"), files))
        return [base + log for log in logs]

    def load_unity_logs(self, file: str):
        """Load unity logs from `file`.

        Args:
            file (str): Path to the unity log file to load.

        Returns:
            pd.DataFrame: The loaded logs.
        """
        with open(file, "r") as f:
            logs = f.readlines()

        header = logs[0].strip().split("\t")
        cols = []
        for h in header:
            if h in ["Timestamp", "Gaze", "Grab"]:
                cols.append(h)
            else:
                cols.append(f"{h}_x")
                cols.append(f"{h}_y")
                cols.append(f"{h}_z")
        rows = list(map(lambda x: x.replace(",", "\t").split("\t"), logs[1:]))
        logs = pd.DataFrame(rows, columns=cols)
        return logs

    def get_conf_mat(self):
        results = self.load_model_eval_metrics()
        confmat = utils.get_conf_mat(
            results, self.config.paths, self.config.gesture_ids
        )
        return confmat


def load_model_eval_metrics(subject, sensor, features, stage, adaptation):
    """Load model evaluation metrics and create the associated `Config` object.

    Args:
        config (Config): Configuration object.
        pre (bool): Load pre-VR metrics post-VR metrics.

    Returns:
        SubjectResults, dict: Subject Results, Evaluation metrics.
    """
    sr = SubjectResults(subject, adaptation, stage, sensor, features)
    return sr, sr.load_model_eval_metrics()


def load_all_model_eval_metrics(
    adaptation: bool,
    pre: bool,
    sensor: EmgSensorType = EmgSensorType.BioArmband,
    features: str | list = "TDPSD",
) -> tuple[list[SubjectResults], list[dict]]:
    stage = ExperimentStage.SG_PRE_TEST if pre else ExperimentStage.SG_POST_TEST
    subjects = sorted(list(filter(lambda d: d.isnumeric(), os.listdir("data/"))))
    configs = []
    metrics = []
    for s in subjects:
        config, metric = load_model_eval_metrics(
            int(s), sensor, features, stage, adaptation
        )
        configs.append(config)
        metrics.append(metric)

    return (configs, metrics)


def get_overall_eval_metrics(results: list[dict]):
    """From a list of results, get the overall evaluation metrics."""
    metrics = {}
    for result in results:
        for k, v in result.items():
            if k == "CONF_MAT":
                v = np.array(v)

            if k not in metrics:
                metrics[k] = [v]
            else:
                metrics[k].append(v)

    return metrics


def get_subjects(base: str):
    """Get all subjects.

    Args:
        base (str): Base directory where subjects are stored.

    Returns:
        list[int]: List of subject IDs.
    """
    subjects = list(filter(lambda d: d.isnumeric(), os.listdir(base)))
    return sorted([int(subject.split("/")[-1]) for subject in subjects])


def main():
    subject = 8
    sensor = EmgSensorType.BioArmband
    features = "TDPSD"
    adaptation = True
    stage = ExperimentStage.SG_PRE_TEST

    sr, results_subj = load_model_eval_metrics(
        subject, sensor, features, stage, adaptation
    )
    # ts, preds, feats = sr.load_predictions()
    memory0 = sr.load_memory(0)
    memory = sr.load_memory(1000)
    print(len(memory0), len(memory))
    outcomes = memory.experience_outcome[len(memory0) :]
    print(
        f"P{sr.config.subject_id} Accuracy: {100 * outcomes.count('P') / len(outcomes)} ({len(outcomes)})"
    )
    print(
        f"dt between memory {np.mean(np.diff(memory.experience_timestamps[len(memory0):]))}"
    )
    exit()
    srs, _ = load_all_model_eval_metrics(False, False)
    for sr in srs:
        # TODO ignore initial training data
        memory0 = sr.load_memory(0)
        memory = sr.load_memory(1000)
        print(len(memory0), len(memory))
        outcomes = memory.experience_outcome[len(memory0) :]
        print(
            f"P{sr.config.subject_id} Accuracy: {100 * outcomes.count('P') / len(outcomes)} ({len(outcomes)})"
        )

    # sr.get_conf_mat()
    # sr.set_stage(ExperimentStage.SG_POST_TEST)
    # sr.get_conf_mat()

    # srs, results = load_all_model_eval_metrics(
    #     True,
    #     True,
    #     EmgSensorType.BioArmband,
    #     "TDPSD",
    # )
    # results = get_overall_eval_metrics(results)
    # utils.get_conf_mat(results, config.paths, config.gesture_ids)

    plt.show()


if __name__ == "__main__":
    log.basicConfig(level=log.INFO)
    # print(main_verify_avg_dt())
    main()
