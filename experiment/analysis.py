from ast import Sub
from multiprocessing import context
import os
from sqlite3 import adapt
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
        memories = list(
            filter(
                lambda x: x.startswith("classifier_memory") and "_ls_" not in x, files
            )
        )
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

    def load_concat_memories(self, ignore_0: bool = True):
        """Load and concatenate all memories. Ignores the memory ID 1000.

        Returns:
            Memory: concatenated memory.
        """
        mems = self.find_memory_ids()
        mems.remove(1000)

        if ignore_0:
            mems.remove(0)

        memory = Memory()
        for mem in mems:
            memory += self.load_memory(mem)
        return memory

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

    def get_experiment_completion_dz(
        self, items=["Apple", "FryingPan", "Key", "ChickenLeg", "Cheery", "SmartPhone"]
    ):
        """Get the completion of the experiment from the logs.

        Args:
            logs (pd.DataFrame): Unity logs.

        Returns:
            float: Completion percentage.
        """
        ulogs = self.load_unity_logs(self.find_unity_logs()[0])
        dt = (
            float(ulogs["Timestamp"].iat[-1]) - float(ulogs["Timestamp"].iat[0])
        ) / 1000
        ret = {"completed": 0, "time": dt, "adap": self.adaptation}

        for i, item in enumerate(items):
            ilogz = ulogs[f"{item}_z"].astype(float).to_numpy()
            dz = np.diff(ilogz)

            ilog = np.abs(np.sum(dz))
            completed = np.any(ilog > 0.5)

            t0, t1, dt = -1, -1, -1
            if completed:
                ret["completed"] = i + 1

                # dz[dz < 0.0001] = 0
                moving = np.nonzero(np.abs(dz) > 0.0001)[0]
                exp_start = float(ulogs["Timestamp"].iat[0]) / 1000
                t0 = float(ulogs["Timestamp"].iat[moving[0]]) / 1000 - exp_start
                t1 = float(ulogs["Timestamp"].iat[moving[-1]]) / 1000 - exp_start
                dt = t1 - t0
                if dt < 0.5:
                    log.warning(
                        f"Subject {self.subject} very fast completion time for {item} (CIIL={self.adaptation}) = {dt=:.3f} s"
                    )
            else:
                log.warning(
                    f"Subject {self.subject} did not complete {item} in CIIL = {self.adaptation}"
                )
            ret[item] = {
                "completed": completed,
                "start": t0,
                "finish": t1,
                "time": dt,
            }
        if ret["completed"] < len(items):
            log.info(
                f"Subject {self.subject} did not complete the experiment CIIL = {self.adaptation} ({ret['completed']} in {ret['time']:.3f} s)"
            )
            ret["time"] = 300
        return ret

    def get_experiment_completion_hand(
        self, items=["Apple", "FryingPan", "Key", "ChickenLeg", "Cheery", "SmartPhone"]
    ):
        """NOT RELIABLE, INSTEAD USE `get_experiment_completion_memory`

        Get the completion of the experiment from the logs, particularly with the hand position vs object.

        Args:
            logs (pd.DataFrame): Unity logs.

        Returns:
            float: Completion percentage.
        """
        ulogs = self.load_unity_logs(self.find_unity_logs()[0])
        T = ulogs["Timestamp"].astype(float).to_numpy() / 1000

        ret = {
            "completed": 0,
            "adap": self.adaptation,
            "subject": self.subject,
        }

        for i, item in enumerate(items):
            t0, t1, dt = 0, 0, 0

            # Check if item was completed
            ilogz = ulogs[f"{item}_z"].astype(float).to_numpy()
            dz = np.diff(ilogz)
            ilog = np.abs(np.sum(dz))
            completed = np.any(ilog > 0.5)

            # If yes, get the time it took to complete
            if completed:
                ret["completed"] = i + 1

                handpos = np.array(
                    [
                        ulogs[f"Hand_{a}"].astype(float).to_numpy()
                        for a in ["x", "y", "z"]
                    ]
                ).T
                objpos = np.array(
                    [
                        ulogs[f"{item}_{a}"].astype(float).to_numpy()
                        for a in ["x", "y", "z"]
                    ]
                ).T

                grabbing = ulogs["Grab"].to_numpy()
                grabbing[grabbing == "False\n"] = 0
                grabbing[grabbing == "True\n"] = 1
                grabbing = grabbing.astype(int)

                distance = np.linalg.norm(handpos - objpos, axis=1)
                distance[grabbing == 0] = 1000

                where_close = np.nonzero(distance < 0.4)
                t_close = T[where_close]

                t_start = T[0]
                try:
                    t0 = (t_close[0]) - t_start
                    t1 = (t_close[-1]) - t_start
                    dt = t1 - t0
                except:
                    log.warning(
                        f"Subject {self.subject} {item} min distance {distance.min()}, {distance.mean()}"
                    )
            else:
                log.warning(
                    f"Subject {self.subject} did not complete {item} in CIIL = {self.adaptation}"
                )
            ret[item] = {
                "completed": completed,
                "start": t0,
                "finish": t1,
                "time": dt,
            }
        if ret["completed"] < len(items):
            log.info(
                f"Subject {self.subject} did not complete the experiment CIIL = {self.adaptation} ({ret['completed']})"
            )
            ret["time"] = 300
        else:
            ret["time"] = np.sum([ret[item]["time"] for item in items])
        return ret

    def get_experiment_completion_memory(
        self, items=["Apple", "FryingPan", "Key", "ChickenLeg", "Cheery", "SmartPhone"]
    ):
        """Get the completion of the experiment from the logs, particularly with the hand position vs object.

        Args:
            logs (pd.DataFrame): Unity logs.

        Returns:
            float: Completion percentage.
        """
        mem = self.load_concat_memories(self.subject != 0)
        ulogs = self.load_unity_logs(self.find_unity_logs()[0])

        t_unity = ulogs["Timestamp"].astype(float).to_numpy() / 1000
        t_unity -= t_unity[0]

        context_diffs = np.diff(np.sum(np.abs(mem.experience_context), axis=1), axis=0)
        context_changes = np.nonzero(context_diffs)

        t_mem = np.array(mem.experience_timestamps)
        t_mem -= t_mem[0]
        n_invalid = np.count_nonzero(t_mem >= 300)
        t_mem = t_mem[: -(n_invalid + 1)]
        mem.experience_outcome = mem.experience_outcome[: -(n_invalid + 1)]

        ret = {
            "completed": 0,
            "adap": self.adaptation,
            "subject": self.subject,
            "precision": mem.experience_outcome.count("P")
            / len(mem.experience_outcome),
        }

        for i, item in enumerate(items):
            t0, t1, dt = 0, 0, 0

            # Check if item was completed
            ilogz = ulogs[f"{item}_z"].astype(float).to_numpy()
            dz = np.diff(ilogz)
            ilog = np.abs(np.sum(dz))
            completed = np.any(ilog > 0.5)

            # If yes, get the time it took to complete
            if completed:
                ret["completed"] = i + 1
                ret["completed"] = i + 1

                # dz[dz < 0.0001] = 0
                moving = np.nonzero(np.abs(dz) > 0.0001)[0]
                exp_start = float(ulogs["Timestamp"].iat[0]) / 1000
                t0 = float(ulogs["Timestamp"].iat[moving[0]]) / 1000 - exp_start
                t1 = float(ulogs["Timestamp"].iat[moving[-1]]) / 1000 - exp_start
                dt = t1 - t0
                if dt < 0.5:
                    log.warning(
                        f"Subject {self.subject} very fast completion time for {item} (CIIL={self.adaptation}) = {dt=:.3f} s"
                    )
            else:
                log.warning(
                    f"Subject {self.subject} did not complete {item} in CIIL = {self.adaptation}"
                )
            ret[item] = {
                "completed": completed,
                "start": t0,
                "finish": t1,
                "time": dt,
            }
        if ret["completed"] < len(items):
            log.info(
                f"Subject {self.subject} did not complete the experiment CIIL = {self.adaptation} ({ret['completed']})"
            )
            ret["time"] = 300
        else:
            # ret["time"] = np.sum([ret[item]["time"] for item in items])
            ret["time"] = t_mem[-1]
        return ret

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
    """From a list of results, get the average evaluation metrics."""
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


def get_dt_logs_vs_memory():
    """Load all subjects and compare the time between their Unity log files and memory.

    Returns:
        list: List of dictionaries {'adapt': {'unity': float, 'memory': float}, 'no_adapt': {'unity': float, 'memory': float}}
    """
    sensor = EmgSensorType.BioArmband
    features = "TDPSD"
    stage = ExperimentStage.SG_POST_TEST
    # Compare memory time vs unity time
    stimes = []
    for subject in [0, 1, 2, 3, 4, 5, 6, 7, 8]:
        times = {
            "adapt": {"unity": 0, "memory": 0},
            "no_adapt": {"unity": 0, "memory": 0},
        }
        for adap in [False, True]:
            sr, results_subj = load_model_eval_metrics(
                subject, sensor, features, stage, adap
            )
            ulogs = sr.load_unity_logs(sr.find_unity_logs()[0])
            uts = ulogs["Timestamp"]
            uts = uts.astype(float)
            uts /= 1000

            m = sr.load_concat_memories(ignore_0=False if subject == 0 else True)

            unity_dt = float(uts.iat[-1]) - float(uts.iat[0])
            memory_dt = m.experience_timestamps[-1] - m.experience_timestamps[0]

            times["adapt" if adap else "no_adapt"]["unity"] = unity_dt
            times["adapt" if adap else "no_adapt"]["memory"] = memory_dt

            # print(
            #     f"{subject=}, {adap=}. Unity dt: {float(uts.iat[-1]) - float(uts.iat[0]):.2f} s, Memory dt: {m.experience_timestamps[-1] - m.experience_timestamps[0]:.2f} s"
            # )
        stimes.append(times)

    for i, time in enumerate(stimes):
        print(f"subject {i}", time)
    return stimes


def analyze_completion(dir="data/"):
    sensor = EmgSensorType.BioArmband
    features = "TDPSD"
    stage = ExperimentStage.SG_POST_TEST

    stats = []
    for subject in get_subjects(dir):
        # print(f"===== {subject=} ======")
        for adap in [False, True]:
            sr = SubjectResults(subject, adap, stage, sensor, features)
            # if sr.subject not in [3]:
            #     continue
            stats.append(sr.get_experiment_completion_memory())
    return stats


def analyze_logs_dir(dir="data/unity/"):
    files = os.listdir(dir)
    logfiles = sorted(list(filter(lambda x: x.startswith("OL"), files)))
    for logf in logfiles:
        with open(dir + logf, "r") as f:
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

            uts = logs["Timestamp"]
            uts = uts.astype(float)
            uts /= 1000
            unity_dt = float(uts.iat[-1]) - float(uts.iat[0])
            print(f"{logf=}, experiment time {unity_dt=:.3f} s")


def main():
    sensor = EmgSensorType.BioArmband
    features = "TDPSD"
    stage = ExperimentStage.SG_POST_TEST

    adaptation = False

    import warnings

    warnings.filterwarnings("ignore")

    completions = analyze_completion()

    completions_na = list(filter(lambda x: not x["adap"], completions))
    completions_ciil = list(filter(lambda x: x["adap"], completions))

    obj_na = np.mean([c["completed"] for c in completions_na])
    obj_ciil = np.mean([c["completed"] for c in completions_ciil])

    prec_na = np.mean([c["precision"] for c in completions_na])
    prec_ciil = np.mean([c["precision"] for c in completions_ciil])

    ct_na = np.mean([comp["time"] for comp in completions_na])
    ct_ciil = np.mean([comp["time"] for comp in completions_ciil])

    score_na = (
        np.mean([c["completed"] * c["precision"] / c["time"] for c in completions_na])
        * 150
        / 6
    )
    score_ciil = (
        np.mean([c["completed"] * c["precision"] / c["time"] for c in completions_ciil])
        * 150
        / 6
    )

    print(f"Average objects NA = {obj_na:.3f}, CIIL = {obj_ciil:.3f}")
    print(f"Average completion time NA = {ct_na:.3f} s, CIIL = {ct_ciil:.3f} s")
    print(f"Average precsision NA = {100*prec_na:.2f} %, CIIL = {100*prec_ciil:.2f} %")
    print(f"Average score NA = {score_na:.3f}, CIIL = {score_ciil:.3f}")

    # TODO cleanup memories that go > 300s
    # get_dt_logs_vs_memory()
    # analyze_logs_dir()
    exit()

    # memory0 = sr.load_memory(0)
    # memory = sr.load_memory(1000)
    # print(len(memory0), len(memory))
    # outcomes = memory.experience_outcome[len(memory0) :]
    # print(
    #     f"P{sr.config.subject_id} Accuracy: {100 * outcomes.count('P') / len(outcomes)} ({len(outcomes)})"
    # )

    # memory0 = sr.load_memory(0)
    # memory = sr.load_memory(1000)
    # print(len(memory0), len(memory))
    # outcomes = memory.experience_outcome[len(memory0) :]
    # print(
    #     f"P{sr.config.subject_id} Accuracy: {100 * outcomes.count('P') / len(outcomes)} ({len(outcomes)})"
    # )
    # print(
    #     f"dt between memory {np.mean(np.diff(memory.experience_timestamps[len(memory0):]))}"
    # )
    # exit()
    # srs, _ = load_all_model_eval_metrics(False, False)

    plt.show()


if __name__ == "__main__":
    # log.basicConfig(level=log.INFO)
    # print(main_verify_avg_dt())
    main()
