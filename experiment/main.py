from lightning.pytorch import seed_everything
import matplotlib.pyplot as plt

from nfc_emg import models, utils
from nfc_emg.sensors import EmgSensorType

from config import Config, ExperimentStage
from familiarization import Familiarization
from game import Game


def main(
    subject_id,
    sensor,
    features,
    step,
    adaptation,
    sample_data,
    negative_method,
    relabel_method,
    powerline_freq,
):
    config = Config(
        subject_id=subject_id,
        sensor_type=sensor,
        features=features,
        stage=step,
        adaptation=adaptation,
        negative_method=negative_method,
        relabel_method=relabel_method,
        powerline_notch_freq=powerline_freq,
    )

    if config.stage == ExperimentStage.FAMILIARIZATION:
        fam = Familiarization(config, False)
        fam.run()
    elif config.stage == ExperimentStage.VISUALIZE_CLASSIFIER:
        fam = Familiarization(config, True)
        fam.run()
    elif config.stage == ExperimentStage.SG_TRAIN:
        models.main_train_nn(
            config.model,
            config.sensor,
            sample_data,
            config.features,
            config.gesture_ids,
            config.paths.gestures,
            config.paths.get_train(),
            config.paths.get_model(),
            config.reps,
            config.rep_time,
        )
    elif config.stage == ExperimentStage.SG_PRE_TEST:
        results = models.main_test_nn(
            config.model,
            config.sensor,
            sample_data,
            config.features,
            config.gesture_ids,
            config.paths.gestures,
            config.paths.get_test(),
        )
        results_file = config.paths.get_results()
        utils.save_eval_results(results, results_file)
        utils.get_conf_mat(results, config.paths, config.gesture_ids)
        plt.show()
    elif config.stage == ExperimentStage.GAME:
        # Change model path for saving after the game
        config.paths.set_model("model_post")
        game = Game(config)
        game.run()
    elif config.stage == ExperimentStage.SG_POST_TEST:
        results = models.main_test_nn(
            config.model,
            config.sensor,
            sample_data,
            config.features,
            config.gesture_ids,
            config.paths.gestures,
            config.paths.get_test(),
        )
        results_file = config.paths.get_results()
        utils.save_eval_results(results, results_file)
        utils.get_conf_mat(results, config.paths, config.gesture_ids)
        plt.show()


if __name__ == "__main__":
    seed_everything(310)
    negative_method = "mixed"  # mixed or none
    # relabel_method = "none"  # LabelSpreading or none
    relabel_method = "LabelSpreading"  # LabelSpreading or none
    features = "TDPSD"
    sensor = EmgSensorType.BioArmband
    mains_freq = 60
    sample_data = False

    # subjects = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    subjects = [99]

    # steps = [ExperimentStage.FAMILIARIZATION]
    # steps = [ExperimentStage.SG_TRAIN, ExperimentStage.SG_PRE_TEST]
    # steps = [ExperimentStage.SG_PRE_TEST]
    # steps = [ExperimentStage.GAME]
    # steps = [ExperimentStage.GAME, ExperimentStage.SG_POST_TEST]
    steps = [ExperimentStage.SG_POST_TEST]
    # steps = [ExperimentStage.SG_PRE_TEST, ExperimentStage.SG_POST_TEST]

    param_1 = False

    for subject in subjects:
        for step in steps:
            message = f"Running {step.name} for {subject}"
            print("=" * len(message))
            print(message)
            print("=" * len(message))

            main(
                subject,
                sensor,
                features,
                step,
                param_1,
                sample_data,
                negative_method,
                relabel_method,
                mains_freq,
            )

    print("Exiting experiment main thread.")
