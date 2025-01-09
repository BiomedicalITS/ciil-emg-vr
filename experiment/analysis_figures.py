import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import logging as log


from memory import Memory
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


def get_avg_completion_time():
    """Get average time between predictions subject-wise. It is normal for memory time to be much longer.
    This is because the classifier still outputs predictions even when there is not object within grasping distance.

    Context is only sent back by Unity when an object is within grasping distance.

    Returns:
        dict: (subject, adaptation) -> (prediction dt, memory dt)
    """
    sensor = EmgSensorType.BioArmband
    features = "TDPSD"
    stage = ExperimentStage.GAME

    dt = {
        "Subject": [],
        "Adaptation": [],
        "Time": [],
    }
    for subject in analysis.get_subjects("data/"):
        for adaptation in [False, True]:
            log.info(f"Loading {subject=}, {adaptation=}")
            sr = analysis.SubjectResults(
                subject,
                adaptation,
                stage,
                sensor,
                features,
            )
            mems = sr.find_memory_ids()
            mems.remove(0)
            mems.remove(1000)

            mem1 = sr.load_memory(mems[0])
            memn = sr.load_memory(mems[-1])

            # print(
            #     np.min(mem1.experience_timestamps), np.max(memn.experience_timestamps)
            # )

            dt["Subject"].append(subject)
            dt["Adaptation"].append(adaptation)
            dt["Time"].append(
                memn.experience_timestamps[-1] - mem1.experience_timestamps[0]
            )

    return pd.DataFrame(dt)


def fix_memory_ts():
    """Get average time between predictions subject-wise. It is normal for memory time to be much longer.
    This is because the classifier still outputs predictions even when there is no object within grasping distance.

    Context is only sent back by Unity when an object is within grasping distance.

    Returns:
        dict: (subject, adaptation) -> (prediction dt, memory dt)
    """
    sensor = EmgSensorType.BioArmband
    features = "TDPSD"
    stage = ExperimentStage.GAME

    for subject in analysis.get_subjects("data/"):
        for adaptation in [False, True]:
            log.info(f"Loading {subject=}")
            sr = analysis.SubjectResults(
                subject,
                adaptation,
                stage,
                sensor,
                features,
            )
            mems = sr.find_memory_ids()
            mems.remove(0)
            mems.remove(1000)

            for i in range(len(mems) - 1):
                mem_valid = sr.load_memory(mems[i])
                mem_under_test = sr.load_memory(mems[i + 1])

                dt = (
                    mem_under_test.experience_timestamps[-1]
                    - mem_valid.experience_timestamps[0]
                )

                if dt < 0:
                    print(
                        f"P{subject}, {adaptation=}: valid adaptation memories end (inclusive) at ID {mems[i]}"
                    )
                    break


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
            memory = Memory()
            for mem in sr.find_memory_ids():
                if mem == 0 or mem == 1000:
                    continue
                memory += sr.load_memory(mem)
            outcomes = memory.experience_outcome
            try:
                if not adap:
                    within_acc_noadap.append(100 * outcomes.count("P") / len(outcomes))
                else:
                    within_acc_adap.append(100 * outcomes.count("P") / len(outcomes))
            except:  # noqa: E722
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
    labels = ["Initial", "Post - None", "Post - P+N"]
    for i, (adap, pre) in enumerate([(False, True), (False, False), (True, False)]):
        _, results = analysis.load_all_model_eval_metrics(adap, pre, sensor, features)
        ca = analysis.get_overall_eval_metrics(results)["CA"]
        ca = [c * 100 for c in ca]

        med = np.median(ca)
        q1 = np.percentile(ca, 25)
        q3 = np.percentile(ca, 75)
        whishi = np.max(ca)
        whislo = np.min(ca)

        print(
            f"{labels[i]}: Lowest {whislo:.2f}, mean ${np.mean(ca):.2f} \pm {np.std(ca):.2f}$~\%, median {med:.2f} "
        )
        stat = {
            "med": med,
            "q1": q1,
            "q3": q3,
            "whislo": 0 if whislo < 0 else whislo,
            "whishi": whishi,
            "mean": np.mean(ca),
            "std": np.std(ca),
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

        # tlx = load_tlx()
        # if i == 1:
        #     tlx = tlx[tlx["Adaptation"] == "N"]
        # elif i == 2:
        #     tlx = tlx[tlx["Adaptation"] == "Y"]
        # else:
        #     continue

        # index = [False, True][i - 1]

        # tlx.drop(columns=["Subject", "Adaptation", "Sensor"], inplace=True)

        # ax.text(
        #     i + 1,
        #     104,
        #     f"NASA-TLX: {tlx.mean(None):.2f} ± {tlx.std().mean(None):.2f}",
        #     ha="center",
        # )
        # ct = get_avg_completion_time().groupby(["Adaptation"])["Time"]
        # ct_mean = ct.mean()
        # ct_std = ct.std()

        # ax.text(
        #     i + 1,
        #     101,
        #     f"Task time: {ct_mean[index]:.2f} ± {ct_std[index]:.2f} s",
        #     ha="center",
        # )

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


def load_tlx(path: str = "data/nfc-emg-experiment - tlx.csv"):
    tlx = pd.read_csv(path)
    return tlx


def print_tlx_table():
    tlx = load_tlx()
    tlx.drop(columns=["Subject", "Sensor"], inplace=True)
    tlx_adap = tlx[tlx["Adaptation"] == "Y"]
    tlx_adap.drop(columns=["Adaptation"], inplace=True)
    tlx_noadap = tlx[tlx["Adaptation"] == "N"]
    tlx_noadap.drop(columns=["Adaptation"], inplace=True)

    for d in tlx_noadap.columns:
        print(
            f"{d} & ${tlx_noadap[d].mean():.2f} \pm {tlx_noadap[d].std():.2f}$ & ${tlx_adap[d].mean():.2f} \pm {tlx_adap[d].std():.2f}$ \\\\"
        )
    print(
        f"Overall & ${tlx_noadap.mean(None):.2f} \pm {np.mean((tlx_noadap.std())):.2f}$ & ${tlx_adap.mean(None):.2f} \pm {np.mean(tlx_adap.std()):.2f}$ \\\\"
    )


if __name__ == "__main__":
    # log.basicConfig(level=log.INFO)
    plt.rcParams.update({"font.size": 32})

    fix_memory_ts()
    # print(get_avg_prediction_dt())
    # ct = get_avg_completion_time()
    # print(ct)
    # print(ct.groupby(["Adaptation"])["Time"].mean())
    # print(ct.groupby(["Adaptation"])["Time"].std())

    # print_tlx_table()

    # fig, axs, _ = pointplot_full()
    # fig.tight_layout()

    # pointplot_pre_post()

    fig, axs, stats = boxplot_pre_post()
    for stat in stats:
        print(stats)
    fig.set_size_inches(16, 9)
    plt.savefig("embc2025/figures/boxplot_pre_post.pdf")

    plt.rcParams.update({"font.size": 15})
    figs = confmat_pre_post()
    names = ["Initial", "None", "P+N"]
    for i, fig in enumerate(figs):
        fig.figure_.tight_layout()
        fig.figure_.savefig(f"embc2025/figures/confmat_pre_post_{names[i]}.png")
    plt.rcParams.update({"font.size": 20})

    # plt.show()
