from lightning.pytorch import seed_everything

from nfc_emg import models, utils
from nfc_emg.sensors import EmgSensorType

from familiarization import Familiarization
from config import Config, ExperimentStage

from game import Game


def main():
    config = Config(
        subject_id="2024_07_11",
        sensor_type=EmgSensorType.BioArmband,
        # stage=ExperimentStage.FAMILIARIZATION,
        # stage=ExperimentStage.VISUALIZE_CLASSIFIER,
        # stage=ExperimentStage.SG_TRAIN,
        # stage=ExperimentStage.SG_TEST,
        stage=ExperimentStage.GAME,
        # stage=ExperimentStage.SG_POST_TEST,
        # adaptation=False,
    )
    # SAMPLE_DATA = False
    SAMPLE_DATA = True

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
            SAMPLE_DATA,
            config.features,
            config.gesture_ids,
            config.paths.gestures,
            config.paths.get_train(),
            config.paths.get_model(),
            config.reps,
            config.rep_time,
        )
    elif config.stage == ExperimentStage.SG_TEST:
        config.paths.test = config.paths.test.replace("test", "pre_test")
        results = models.main_test_nn(
            config.model,
            config.sensor,
            SAMPLE_DATA,
            config.features,
            config.gesture_ids,
            config.paths.gestures,
            config.paths.get_test(),
        )
        results_file = config.paths.get_results().replace(".csv", "_pre.json")
        utils.save_eval_results(results, results_file)
        utils.show_conf_mat(results, config.paths, config.gesture_ids)
    elif config.stage == ExperimentStage.GAME:
        config.paths.set_model("model_post")
        game = Game(config)
        game.run()
    elif config.stage == ExperimentStage.SG_POST_TEST:
        config.paths.test = config.paths.test.replace("test", "post_test")
        results = models.main_test_nn(
            config.model,
            config.sensor,
            True,
            config.features,
            config.gesture_ids,
            config.paths.gestures,
            config.paths.get_fine(),
        )
        results_file = config.paths.get_results().replace(".csv", "_post.json")
        utils.save_eval_results(results, results_file)
        utils.show_conf_mat(results, config.paths, config.gesture_ids)


if __name__ == "__main__":
    seed_everything(310)
    main()
    print("Exiting experiment main thread.")
