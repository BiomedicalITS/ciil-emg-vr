from threading import Lock
import time
import csv

import numpy as np

from libemg.emg_classifier import OnlineEMGClassifier
from libemg.feature_extractor import FeatureExtractor
from libemg.utils import get_windows


def run_classifier(oclassi: OnlineEMGClassifier, save_path: str, lock: Lock):
    """
    Adapted copy-paste of OnlineEMGClassifier._run_helper with some optimizations and saves predictions to a file
    """
    print("SuperClassifier is started!")
    fe = FeatureExtractor()
    oclassi.raw_data.reset_emg()
    with open(save_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile, delimiter=",")
        while True:
            if len(oclassi.raw_data.get_emg()) < oclassi.window_size:
                continue

            time_stamp = time.time()
            data = oclassi._get_data_helper()

            # Extract window and predict sample
            window = get_windows(
                data[-oclassi.window_size :][:],
                oclassi.window_size,
                oclassi.window_size,
            )

            # Dealing with the case for CNNs when no features are used
            features = fe.extract_features(
                oclassi.features, window, oclassi.classifier.feature_params
            )
            # If extracted features has an error - give error message
            if fe.check_features(features) != 0:
                oclassi.raw_data.adjust_increment(
                    oclassi.window_size, oclassi.window_increment
                )
                continue
            classifier_input = oclassi._format_data_sample(features)

            oclassi.raw_data.adjust_increment(
                oclassi.window_size, oclassi.window_increment
            )

            lock.acquire()
            probabilities = oclassi.classifier.classifier.predict_proba(
                classifier_input
            )
            lock.release()

            prediction, probability = oclassi.classifier._prediction_helper(
                probabilities
            )
            prediction = prediction[0]
            probability = probability[0]

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

            oclassi.sock.sendto(bytes(message, "utf-8"), (oclassi.ip, oclassi.port))
            writer.writerow([time_stamp, prediction])
            csvfile.flush()

            if oclassi.std_out:
                print(message)

            # print(f"Classification time: {1000*(time.time() - time_stamp):.3f} ms")
