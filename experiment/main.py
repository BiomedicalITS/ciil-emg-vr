from lightning.pytorch import seed_everything

from nfc_emg import models
from nfc_emg.sensors import EmgSensorType

from familiarization import Familiarization
from config import Config, ExperimentStage

from game import Game


def main():
    config = Config(
        subject_id=0,
        sensor_type=EmgSensorType.BioArmband,
        adaptation=True,
        stage=ExperimentStage.SG_TRAIN,
    )

    if config.stage == ExperimentStage.FAMILIARIZATION:
        fam = Familiarization(config)
        fam.run()
    elif config.stage == ExperimentStage.SG_TRAIN:
        models.main_train_nn(
            config.model,
            config.sensor,
            True,
            config.features,
            config.gesture_ids,
            config.paths.gestures,
            config.paths.get_train(),
            config.paths.get_model(),
            config.reps,
            config.rep_time,
        )
    elif config.stage == ExperimentStage.SG_TEST:
        models.main_test_nn(
            config.model,
            config.sensor,
            True,
            config.features,
            config.gesture_ids,
            config.paths.gestures,
            config.paths.get_test(),
        )
    elif config.stage == ExperimentStage.GAME:
        print("TODO: should you ditch old data after a training pass???")
        print("TODO: save training pass results?")
        print("TODO: save Unity logs to disk")
        config.paths.set_model("model_post")
        game = Game(config)
        game.run()
    elif config.stage == ExperimentStage.POST_SG_TEST:
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
