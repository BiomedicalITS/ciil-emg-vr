import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging as log


from nfc_emg.sensors import EmgSensorType
from nfc_emg import utils

from experiment import analysis
from experiment.config import ExperimentStage


def get_avg_prediction_dt():
    """Get average time between predictions subject-wise. It is normal for memory time to be much longer.
    This is because the classifier still outputs predictions even when there is not object within grasping distance.

    Context is only sent back by Unity when an object is within grasping distance.

    Returns:
        dict: (subject, adaptation) -> (prediction dt, memory dt)
    """
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

            len0 = len(sr.load_memory(0).experience_outcome)
            mem = sr.load_memory(1000).experience_timestamps[len0:]

            dt[(subject, adaptation)] = (
                np.mean(np.diff(ts)).item(0),
                np.mean(np.diff(mem)).item(0),
            )
            log.info(
                f"Number of predictions: {len(ts)}, number of memories: {len(mem)}"
            )
            log.info(
                f"Prediction dt={1000*dt[(subject, adaptation)][0]:.3f} ms, Memory dt={1000*dt[(subject, adaptation)][1]:.3f} ms"
            )
    return dt


def pointplot_pre_post(sensor=EmgSensorType.BioArmband, features="TDPSD"):
    """Create a boxplot figure with three boxes: initial test, post-test without adaptation, post-test with adaptation.

    Args:
        sensor (_type_, optional): Sensor used. Defaults to EmgSensorType.BioArmband.
        features (str, optional): Feature set used. Defaults to "TDPSD".

    Returns:
        tuple: fig, ax, stats
    """
    stats = []
    participants = []
    labels = ["Initial", "Post"]
    for i, (adap, pre) in enumerate([(False, True), (False, False), (True, False)]):
        srs, results = analysis.load_all_model_eval_metrics(adap, pre, sensor, features)
        ca = analysis.get_overall_eval_metrics(results)["CA"]
        ca = [c * 100 for c in ca]

        for sr in srs:
            participants.append(sr.config.subject_id)

        stats.append(ca)

    # TODO participant # is broken like this
    fig, ax = plt.subplots(figsize=(16, 9))
    colors = list(mcolors.TABLEAU_COLORS.values())
    for subject in range(len(stats[0])):
        color = colors[subject % len(colors)]
        ax.plot(
            labels,
            [stats[0][subject], stats[1][subject]],
            marker="o",
            label=f"P{participants[subject]}, no adaptation",
            # linestyle="dashed",
            color=color,
        )
        ax.plot(
            labels,
            [stats[0][subject], stats[2][subject]],
            marker="o",
            linestyle="dashed",
            label=f"P{participants[subject]}, with adaptation",
            color=color,
        )
    ax.legend()
    ax.set_xlabel("Test case")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.grid(True)

    return fig, ax, stats


def pointplot_full(sensor=EmgSensorType.BioArmband, features="TDPSD"):
    """Create a boxplot figure with three boxes: initial test, post-test without adaptation, post-test with adaptation.

    Args:
        sensor (_type_, optional): Sensor used. Defaults to EmgSensorType.BioArmband.
        features (str, optional): Feature set used. Defaults to "TDPSD".

    Returns:
        tuple: fig, ax, stats
    """
    stats = []
    participants = []
    within_acc_noadap = []
    within_acc_adap = []
    for i, (adap, pre) in enumerate([(False, True), (False, False), (True, False)]):
        srs, results = analysis.load_all_model_eval_metrics(adap, pre, sensor, features)
        ca = analysis.get_overall_eval_metrics(results)["CA"]
        ca = [c * 100 for c in ca]

        for sr in srs:
            participants.append(sr.config.subject_id)
            mem0 = len(sr.load_memory(0))
            mem = sr.load_memory(-1)
            outcomes = mem.experience_outcome[mem0:]

            try:
                if not adap:
                    within_acc_noadap.append(100 * outcomes.count("P") / len(outcomes))
                else:
                    within_acc_adap.append(100 * outcomes.count("P") / len(outcomes))
            except:
                print(f"Error in P{sr.config.subject_id}")
                if not adap:
                    within_acc_noadap.append(0)
                else:
                    within_acc_adap.append(0)
        stats.append(ca)

    labels = ["Initial", "Online", "Post"]
    fig, ax = plt.subplots(figsize=(16, 9))
    colors = list(mcolors.TABLEAU_COLORS.values())
    for subject in range(len(stats[0])):
        color = colors[subject % len(colors)]
        ax.plot(
            labels,
            [stats[0][subject], within_acc_noadap[subject], stats[1][subject]],
            marker="o",
            label=f"P{participants[subject]}, no adaptation",
            # linestyle="dashed",
            color=color,
        )
        ax.plot(
            labels,
            [stats[0][subject], within_acc_adap[subject], stats[2][subject]],
            marker="o",
            linestyle="dashed",
            label=f"P{participants[subject]}, with adaptation",
            color=color,
        )
    ax.legend()
    ax.set_xlabel("Test case")
    ax.set_ylabel("Classification Accuracy (%)")
    ax.grid(True)

    return fig, ax, stats


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
    log.basicConfig(level=log.INFO)
    # print(get_avg_prediction_dt())
    # pointplot_pre_post()
    pointplot_full()
    # boxplot_pre_post()
    # confmat_pre_post()
    plt.show()
