from threading import Lock
import time
import csv

import numpy as np

from libemg.emg_classifier import OnlineEMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows


def run_classifier(oclassi: OnlineEMGClassifier, save_path: str, lock: Lock):
    """
    Adapted copy-paste of OnlineEMGClassifier._run_helper.

    Therefore, `oclassi.run()` should not be used.

    Essentially, it:

    - Waits for enough new EMG data.
    - Calculates the features.
    - Does a prediction with the features.
    - Saves the predictions to a file and sends it to its UDP socket.
    """
    print("SuperClassifier is started!")
    fe = FeatureExtractor()
    oclassi.raw_data.reset_emg()
    with open(save_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        while True:
            if oclassi.raw_data.get_emg() is None:
                time.sleep(0.005)
                continue
            if len(oclassi.raw_data.get_emg()) < oclassi.window_size:
                time.sleep(0.005)
                continue

            time_stamp = f"{time.perf_counter():.4f}"
            data = oclassi._get_data_helper()

            # Extract window and predict sample
            # (n_windows, n_emg_ch, ws)
            window = get_windows(
                data[-oclassi.window_size :][:],
                oclassi.window_size,
                oclassi.window_size,  # why???
            )

            # dict: {feature_name: np.array of shape (n_windows, n_features)}
            features = fe.extract_features(
                oclassi.features, window, oclassi.classifier.feature_params, array=True
            )

            # If extracted features has an error - give error message
            if fe.check_features(features) != 0:
                oclassi.raw_data.adjust_increment(
                    oclassi.window_size, oclassi.window_increment
                )
                continue
            # classifier_input = oclassi._format_data_sample(features)
            classifier_input = features  # test

            oclassi.raw_data.adjust_increment(
                oclassi.window_size, oclassi.window_increment
            )

            with lock:
                probabilities = oclassi.classifier.classifier.predict_proba(
                    classifier_input
                )

            prediction, probability = oclassi.classifier._prediction_helper(
                probabilities
            )
            prediction = prediction[0]
            probability = probability[0]

            # Don't take into account post-processing for csv
            newline = [time_stamp, prediction] + features.flatten().tolist()
            writer.writerow(newline)
            csvfile.flush()

            # print(
            #     f"({time.time():.4f}) classifier: Wrote 1 new lines (total {preds_count})"
            # )

            # Check for rejection
            if oclassi.classifier.rejection:
                # TODO: Right now this will default to -1
                prediction = oclassi.classifier._rejection_helper(
                    prediction, probability
                )
            oclassi.previous_predictions.append(prediction)

            # Check for majority vote
            if oclassi.classifier.majority_vote:
                values, counts = np.unique(
                    list(oclassi.previous_predictions), return_counts=True
                )
                prediction = values[np.argmax(counts)]
            message = f"{prediction} {time_stamp}"

            # print(message
            time.sleep(0.003)
            oclassi.sock.sendto(message.encode(), (oclassi.ip, oclassi.port))

            if oclassi.std_out:
                print(message)

            # print(
            #     f"Classification time: {1000*(time.perf_counter() - float(time_stamp)):.3f} ms"
            # )
