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
        stage=ExperimentStage.GAME,
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
            config.paths.train,
            config.paths.model,
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
            config.paths.test,
        )
    elif config.stage == ExperimentStage.GAME:
        print("TODO: NfcPaths create new folders for memory and models")
        print("TODO: should you ditch old data after a training pass???")
        print("TODO: save training pass results?")
        config.paths.set_model_name("model_post.pth")
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
            config.paths.fine,
        )


if __name__ == "__main__":
    seed_everything(310)
    main()

    print("Exiting experiment main thread.")
