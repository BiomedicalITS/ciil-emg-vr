import numpy as np
import matplotlib.pyplot as plt
import logging as log


from nfc_emg.sensors import EmgSensorType
from nfc_emg import utils

from experiment import analysis
from experiment.config import ExperimentStage


def get_avg_prediction_dt():
    sensor = EmgSensorType.BioArmband
    features = "TDPSD"
    stage = ExperimentStage.GAME

    dt = {}
    for subject in analysis.get_subjects("data/"):
        for adaptation in [True, False]:
            log.info(f"Loading {subject=}, {adaptation=}")
            sr = analysis.SubjectResults(
                subject,
                adaptation,
                stage,
                sensor,
                features,
            )
            ts, _, _ = sr.load_predictions()
            dt[(subject, adaptation)] = np.mean(np.diff(ts)).item(0)
            log.info(f"dt {dt[(subject, adaptation)]}")
    return dt


def boxplot_pre_post(sensor=EmgSensorType.BioArmband, features="TDPSD"):
    """Create a boxplot figure with three boxes: initial test, post-test without adaptation, post-test with adaptation.

    Args:
        sensor (_type_, optional): Sensor used. Defaults to EmgSensorType.BioArmband.
        features (str, optional): Feature set used. Defaults to "TDPSD".

    Returns:
        tuple: fig, ax, stats
    """
    stats = []
    labels = ["Initial", "Post w/o", "Post w/"]
    for i, (adap, pre) in enumerate([(False, True), (False, False), (True, False)]):
        _, results = analysis.load_all_model_eval_metrics(adap, pre, sensor, features)
        ca = analysis.get_overall_eval_metrics(results)["CA"]
        ca = [c * 100 for c in ca]

        med = np.median(ca)
        q1 = np.percentile(ca, 25)
        q3 = np.percentile(ca, 75)
        whishi = np.max(ca)
        whislo = np.min(ca)

        stat = {
            "med": med,
            "q1": q1,
            "q3": q3,
            "whislo": 0 if whislo < 0 else whislo,
            "whishi": whishi,
            "mean": np.mean(ca),
            "label": labels[i],
            "len": len(ca),
        }
        stats.append(stat)

    fig, ax = plt.subplots(figsize=(16, 9))
    bplot = ax.bxp(
        stats, showfliers=False, showmeans=True, meanline=True, patch_artist=True
    )

    for i, stat in enumerate(stats):
        bplot["boxes"][i].set_facecolor("lightblue")

    ax.set_xlabel("Test case")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.grid(True)

    return fig, ax, stats


def confmat_pre_post(sensor=EmgSensorType.BioArmband, features="TDPSD"):
    """Show confusion matrices for pre-test, post-test without adaptation, post-test with adaptation.

    Args:
        sensor (_type_, optional): _description_. Defaults to EmgSensorType.BioArmband.
        features (str, optional): _description_. Defaults to "TDPSD".

    Returns:
        list[ConfusionMatrixDisplay]: list of figures
    """
    figs = []
    for adap, pre in [(False, True), (False, False), (True, False)]:
        srs, results = analysis.load_all_model_eval_metrics(adap, pre, sensor, features)
        metrics = analysis.get_overall_eval_metrics(results)

        paths = srs[0].config.paths
        gest_ids = srs[0].config.gesture_ids
        confmats = np.sum(metrics["CONF_MAT"], axis=0) / len(metrics["CONF_MAT"])
        figs.append(utils.get_conf_mat({"CONF_MAT": confmats}, paths, gest_ids))
    return figs


if __name__ == "__main__":
    boxplot_pre_post()
    confmat_pre_post()
    plt.show()
