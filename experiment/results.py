import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json

from config import Config, ExperimentStage
from nfc_emg.sensors import EmgSensorType


def find_unity_logs(dir: str):
    """Find unity logs in `dir`.

    Args:
        dir (str): Directory in which to log for logs

    Returns:
        list[str]: List of log files found.
    """
    files = os.listdir(dir)
    logs = list(filter(lambda x: x.startswith("OL"), files))
    return [dir + log for log in logs]


def load_unity_logs(file: str):
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


def load_model_eval_metrics(config: Config, pre: bool):
    """Load model evaluation metrics from `config` internal paths.

    Args:
        config (Config): Configuration object.
        pre (bool): Load pre-VR metrics post-VR metrics.

    Returns:
        json: Evaluation metrics.
    """
    with open(
        config.paths.get_experiment_dir() + f"results_{'pre' if pre else 'post'}.json",
        "r",
    ) as f:
        return json.load(f)


def load_all_model_eval_metrics(
    adaptation: bool,
    pre: bool,
    sensor: EmgSensorType = EmgSensorType.BioArmband,
    features: str | list = "TDPSD",
):
    subjects = list(filter(lambda d: d.isnumeric(), os.listdir("data/")))
    metrics = []
    for s in subjects:
        config = Config(
            subject_id=int(s),
            sensor_type=sensor,
            features=features,
            stage=ExperimentStage.SG_PRE_TEST if pre else ExperimentStage.SG_POST_TEST,
            adaptation=adaptation,
        )
        metrics.append(load_model_eval_metrics(config, pre))

    return metrics


def __main():
    config = Config(
        subject_id=0,
        sensor_type=EmgSensorType.BioArmband,
        features="TDPSD",
        stage=ExperimentStage.GAME,
        adaptation=True,
    )

    logfiles = find_unity_logs(config.paths.get_experiment_dir())
    logs = load_unity_logs(logfiles[0])
    print(logs.head())

    print(load_model_eval_metrics(config, True))
    print(
        len(
            load_all_model_eval_metrics(
                True,
                True,
                EmgSensorType.BioArmband,
                "TDPSD",
            )
        )
    )


if __name__ == "__main__":
    __main()
