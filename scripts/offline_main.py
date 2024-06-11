import torch

from nfc_emg import utils, datasets, models
from nfc_emg.models import EmgSCNN, EmgCNN, CosineSimilarity
from nfc_emg.sensors import EmgSensor, EmgSensorType


def main_scnn(
    sensor: EmgSensor,
    sample_data: bool,
    gestures_list: list,
    gestures_img_dir: str,
    out_data_dir: str,
    model_out_path: str,
):
    if sample_data:
        sensor.start_streamer()
        odh = utils.get_online_data_handler(
            sensor.fs,
            bandpass_freqs=sensor.bandpass_freqs,
            notch_freq=sensor.notch_freq,
            imu=False,
        )
        utils.screen_guided_training(
            odh, gestures_list, gestures_img_dir, 5, 5, out_data_dir
        )

    reps = utils.get_reps(out_data_dir)
    if len(reps) == 1:
        train_reps = reps
        test_reps = []
    else:
        train_reps = reps[: int(0.8 * len(reps))]
        test_reps = reps[int(0.8 * len(reps)) :]

    print("Training on reps:", train_reps)
    print("Testing on reps:", test_reps)

    model = EmgSCNN(sensor.emg_shape, CosineSimilarity())
    model = models.train_model_scnn(
        model,
        sensor,
        out_data_dir,
        gestures_list,
        train_reps,
        test_reps,
        2000,
    )
    torch.save(model.state_dict(), model_out_path)
    return model


def main_cnn(
    device: str,
    sample_data: bool,
    gestures_list: list,
    gestures_img_dir: str,
    out_data_dir: str,
    finetune: bool,
    emg_shape: tuple,
    emg_fs: int,
    emg_notch_freq: int,
    emg_moving_avg_n: int,
    model_out_path: str,
):
    if sample_data:
        utils.setup_streamer(device, emg_notch_freq)
        odh = utils.get_online_data_handler(
            emg_fs, notch_freq=emg_notch_freq, imu=False
        )
        datasets.screen_guided_training(
            odh, gestures_list, gestures_img_dir, 1 if finetune else 5, 5, out_data_dir
        )

    reps = utils.get_reps(out_data_dir)
    if len(reps) == 1:
        train_reps = reps
        test_reps = []
    else:
        train_reps = reps[: int(0.8 * len(reps))]
        test_reps = reps[int(0.8 * len(reps)) :]

    print("Training on reps:", train_reps)
    print("Testing on reps:", test_reps)

    model = (
        EmgCNN(emg_shape, len(gestures_list))
        if not finetune
        else models.get_model(model_out_path, emg_shape, len(gestures_list), True)
    )
    model = models.train_model(
        model,
        out_data_dir,
        train_reps,
        test_reps,
        device,
        emg_moving_avg_n,
    )
    torch.save(model.state_dict(), model_out_path)
    return model


if __name__ == "__main__":
    import configs as g
    import matplotlib.pyplot as plt
    from sklearn.metrics import accuracy_score

    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.metrics import ConfusionMatrixDisplay

    from emager_py import majority_vote as mv

    utils.set_paths(g.DEVICE)
    train_dir, _, model_path, gestures_dir = utils.get_paths()

    SAMPLE_DATA = False
    FINETUNE = False
    TRAIN_GESTURE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 26, 30]  # initial training
    # TEST_GESTURE_IDS = [1, 2, 3, 14, 26, 30]
    TEST_GESTURE_IDS = [1, 2, 3, 4, 5, 6, 7, 8, 14, 15, 30]
    test_gesture_names = [
        utils.map_gid_to_name(gestures_dir)[i] for i in TEST_GESTURE_IDS
    ]

    sensor = EmgSensor(EmgSensorType.MyoArmband)
    sensor.set_majority_vote(100)

    model = models.get_model_scnn(model_path, sensor.emg_shape)
    # model = main_scnn(
    #     sensor=sensor,
    #     sample_data=SAMPLE_DATA,
    #     gestures_list=TRAIN_GESTURE_IDS,
    #     gestures_img_dir=gestures_dir,
    #     out_data_dir=train_dir,
    #     model_out_path=model_path,
    # )
    model.to(g.ACCELERATOR)

    test_gestures_cid = utils.get_cid_from_gid(
        gestures_dir, train_dir, TEST_GESTURE_IDS
    )
    all_reps = utils.get_reps(train_dir)
    odh = datasets.get_offline_datahandler(
        train_dir,
        test_gestures_cid,
        all_reps,
    )
    calib_odh = odh.isolate_data("reps", all_reps[-2:-1])
    calib_data, calib_labels = datasets.prepare_data(calib_odh, sensor, 1, 1)
    calib_data = torch.from_numpy(calib_data).to(g.ACCELERATOR)
    calib_embeds = model(calib_data).detach().cpu().numpy()

    test_odh = odh.isolate_data("reps", all_reps[-1:])
    test_data, test_labels = datasets.prepare_data(test_odh, sensor, 1, 1)

    classifier = LinearDiscriminantAnalysis()
    model.attach_classifier(classifier)
    model.fit_classifier(calib_embeds, calib_labels)
    preds = model.predict(torch.from_numpy(test_data).to(g.ACCELERATOR))
    preds_maj = mv.majority_vote(preds, sensor.maj_vote_n)
    print("*" * 80)
    print(accuracy_score(test_labels, preds))
    print(accuracy_score(test_labels, preds_maj))
    fig = ConfusionMatrixDisplay.from_predictions(
        test_labels, preds_maj, display_labels=test_gesture_names, normalize="true"
    )
    plt.show()
    # test

    # main_cnn(
    #     device=g.DEVICE,
    #     sample_data=SAMPLE_DATA,
    #     gestures_list=g.LIBEMG_GESTURE_IDS,
    #     gestures_img_dir=g.LIBEMG_GESTURES_DIR,
    #     out_data_dir=g.TRAIN_DATA_DIR,
    #     finetune=FINETUNE,
    #     emg_shape=g.EMG_DATA_SHAPE,
    #     emg_fs=g.EMG_SAMPLING_RATE,
    #     emg_notch_freq=g.EMG_NOTCH_FREQ,
    #     emg_moving_avg_n=g.EMG_RUNNING_MEAN_LEN,
    #     model_out_path=g.MODEL_PATH,
    # )
