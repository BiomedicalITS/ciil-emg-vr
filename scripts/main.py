from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from nfc_emg import utils, datasets, models
from nfc_emg.models import EmgSCNN, EmgCNN, CosineSimilarity, EmgSCNNWrapper
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths

import offline_main
import online_main
import configs as c

if __name__ == "__main__":
    subject = 0
    sensor_type = EmgSensorType.BioArmband

    # TODO do both Armband and BioArmband at the same time
    sensor = EmgSensor(sensor_type)
    paths = NfcPaths(f"data/{subject}/{sensor_type.value}/")
    gesture_ids = c.FUNCTIONAL_SET
    accelerator = "cpu"

    mw = offline_main.main_train_scnn(
        sensor,
        paths.train,
        True,
        gesture_ids,
        paths.gestures,
        CosineSimilarity(),
        paths.model,
    )
    mw.save_to_disk(paths.model)

    odw = online_main.OnlineDataWrapper(
        sensor,
        mw,
        paths,
        gesture_ids,
        False,
        c.PSEUDO_LABELS_PORT,
        c.PREDS_PORT,
        c.PREDS_IP,
    )
    odw.run(60)

    mw.save_to_disk(paths.set_model_name("model_post.pth"))

    # TODO short SGT and test with both original model and finetuned model
    utils.do_sgt(sensor, gesture_ids, paths.gestures, paths.test, 2, 3)

    # Now test and save results
    mw_pre = EmgSCNNWrapper.load_from_disk(paths.set_model_name("model_post.pth"))
    models.main_test_scnn(
        mw_pre, sensor, paths.test, False, gesture_ids, paths.gestures
    )
    models.main_test_scnn(mw, sensor, paths.test, False, gesture_ids, paths.gestures)
