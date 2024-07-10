from lightning.pytorch import seed_everything
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

from nfc_emg import models, utils
from nfc_emg.sensors import EmgSensorType

from familiarization import Familiarization
from config import Config, ExperimentStage

from game import Game


def main():
    config = Config(
        subject_id=0,
        sensor_type=EmgSensorType.BioArmband,
        adaptation=True,
        # adaptation=True,
        # stage=ExperimentStage.FAMILIARIZATION,
        # stage=ExperimentStage.VISUALIZE_CLASSIFIER,
        # stage=ExperimentStage.SG_TRAIN,
        stage=ExperimentStage.SG_TEST,
        # stage=ExperimentStage.GAME,
        # stage=ExperimentStage.SG_POST_TEST,
    )
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
        results = models.main_test_nn(
            config.model,
            config.sensor,
            SAMPLE_DATA,
            config.features,
            config.gesture_ids,
            config.paths.gestures,
            config.paths.get_test(),
        )
        conf_mat = results["CONF_MAT"] / np.sum(
            results["CONF_MAT"], axis=1, keepdims=True
        )
        test_gesture_names = utils.get_name_from_gid(
            config.paths.gestures, config.paths.get_train(), config.gesture_ids
        )

        utils.save_eval_results(
            results, config.paths.get_results().replace(".csv", "_pre.json")
        )
        ConfusionMatrixDisplay(conf_mat, display_labels=test_gesture_names).plot()
        plt.show()
    elif config.stage == ExperimentStage.GAME:
        print("TODO: save Unity logs to disk")
        print("TODO: model overfits massively on examples")
        config.paths.set_model("model_post")
        game = Game(config)
        game.run()
    elif config.stage == ExperimentStage.SG_POST_TEST:
        models.main_test_nn(
            config.model,
            config.sensor,
            True,
            config.features,
            config.gesture_ids,
            config.paths.gestures,
            config.paths.get_fine(),
        )


if __name__ == "__main__":
    seed_everything(310)
    main()
    print("Exiting experiment main thread.")
