from re import L

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from nfc_emg import utils, datasets, models
from nfc_emg.models import EmgSCNN, EmgCNN, CosineSimilarity
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths

import offline_main
import online_main
import configs as c

if __name__ == "__main__":
    subject = 0
    sensor_type = EmgSensorType.MyoArmband

    sensor = EmgSensor(sensor_type)
    paths = NfcPaths(f"data/{subject}/{sensor_type.value}")
    gesture_ids = [1, 2, 3, 4, 5, 8, 14, 26, 30]
    accelerator = c.ACCELERATOR

    model = offline_main.main_train_scnn(
        sensor,
        paths.train,
        True,
        gesture_ids,
        paths.gestures,
        LinearDiscriminantAnalysis(),
        paths.model,
    ).to(accelerator)

    odw = online_main.OnlineDataWrapper(sensor, model, paths, gesture_ids, accelerator)
    odw.run()
    models.save_scnn(model, paths.model.replace("model", "model_post"))
