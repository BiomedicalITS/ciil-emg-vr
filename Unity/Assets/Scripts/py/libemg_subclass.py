# libemg_subclass.py
import libemg
import PrepareClassifierEMaGer
import numpy as np
import time
from  libemg.emg_classifier import *
from collections import deque
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from libemg.feature_extractor import FeatureExtractor
from multiprocessing import Process
import numpy as np
import socket
import matplotlib.pyplot as plt
import time
from scipy import stats

from libemg.utils import get_windows


USER_ID = 2
WINDOW_SIZE = 40 #Window size in samples from SGT
WINDOW_INCREMENT = 20
FEATURE_SET = "LS4" #Feature set selection

def start_live_classifier(offline_classifier):

    libemg.streamers.emager_streamer() #process to start myo giving out data
    odh = libemg.data_handler.OnlineDataHandler() #online data handler: process to start grabbing myo data
    odh.start_listening()

    fe = libemg.feature_extractor.FeatureExtractor()

    feature_list = fe.get_feature_groups()[FEATURE_SET]
    classifier = OnlineEMGClassifierUnity(offline_classifier, window_size=WINDOW_SIZE, window_increment=WINDOW_INCREMENT, 
                    online_data_handler=odh, features=feature_list, std_out=True)
    classifier.run(block=True)
    
class OnlineEMGClassifierUnity:
    """OnlineEMGClassifier.

    Given a EMGClassifier and additional information, this class will stream class predictions over UDP in real-time.

    Parameters
    ----------
    offline_classifier: EMGClassifier
        An EMGClassifier object. 
    window_size: int
        The number of samples in a window. 
    window_increment: int
        The number of samples that advances before next window.
    online_data_handler: OnlineDataHandler
        An online data handler object.
    features: list or None
        A list of features that will be extracted during real-time classification. These should be the 
        same list used to train the model. Pass in None if using the raw data (this is primarily for CNNs).
    parameters: dict (optional)
        A dictionary including all of the parameters for the sklearn models. These parameters should match those found 
        in the sklearn docs for the given model.
    port: int (optional), default = 12346
        The port used for streaming predictions over UDP.
    ip: string (optional), default = '127.0.0.1'
        The ip used for streaming predictions over UDP.
    velocity: bool (optional), default = False
        If True, the classifier will output an associated velocity (used for velocity/proportional based control).
    std_out: bool (optional), default = False
        If True, prints predictions to std_out.
    tcp: bool (optional), default = False
        If True, will stream predictions over TCP instead of UDP.
    """
    def __init__(self, offline_classifier, window_size, window_increment, online_data_handler, features, port=12346, ip='127.0.0.1', std_out=False, tcp=False):
        self.window_size = window_size
        self.window_increment = window_increment
        self.raw_data = online_data_handler.raw_data
        self.filters = online_data_handler.fi
        self.features = features
        self.port = port
        self.ip = ip
        self.classifier = offline_classifier

        self.tcp = tcp
        if not tcp:
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        else:
            print("Waiting for TCP connection...")
            self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            self.sock.bind((ip, port))
            self.sock.listen()
            self.conn, addr = self.sock.accept()
            print(f"Connected by {addr}")

        self.process = Process(target=self._run_helper, daemon=True,)
        self.std_out = std_out
        self.previous_predictions = deque(maxlen=self.classifier.majority_vote)


    def run(self, block=True):
        """Runs the classifier - continuously streams predictions over UDP.

        Parameters
        ----------
        block: bool (optional), default = True
            If True, the run function blocks the main thread. Otherwise it runs in a 
            seperate process.
        """
        if block:
            self._run_helper()
        else:
            self.process.start()

    def stop_running(self):
        """Kills the process streaming classification decisions.
        """
        self.process.terminate()

    def _run_helper(self):
        # TODO: enable deep learning classifiers that don't operate on features
        fe = FeatureExtractor()
        self.raw_data.reset_emg()
        while True:
            data = self._get_data_helper()
            if len(data) >= self.window_size:
                # Extract window and predict sample
                window = get_windows(data, self.window_size, self.window_size)

                # Dealing with the case for CNNs when no features are used
                if self.features:
                    features = fe.extract_features(self.features, window, self.classifier.feature_params)
                    # If extracted features has an error - give error message
                    if (fe.check_features(features) != 0):
                        self.raw_data.adjust_increment(self.window_size, self.window_increment)
                        continue
                    classifier_input = self._format_data_sample(features)
                else:
                    classifier_input = window
                self.raw_data.adjust_increment(self.window_size, self.window_increment)
                prediction, probability = self.classifier._prediction_helper(self.classifier.classifier.predict_proba(classifier_input))
                prediction = prediction[0]
                probability = probability[0]

                # Check for rejection
                if self.classifier.rejection:
                    #TODO: Right now this will default to -1
                    prediction = self.classifier._rejection_helper(prediction, probability)
                self.previous_predictions.append(prediction)
                
                # Check for majority vote
                if self.classifier.majority_vote:
                    values, counts = np.unique(list(self.previous_predictions), return_counts=True)
                    prediction = values[np.argmax(counts)]
                
                # Check for velocity based control
                calculated_velocity = ""
                if self.classifier.velocity:
                    calculated_velocity = " 0"
                    # Dont check if rejected 
                    if prediction >= 0:
                        calculated_velocity = " " + str(self.classifier._get_velocity(window, prediction))

                mean_data = np.mean(np.absolute(self.raw_data.get_emg()), axis=0)
                new_min = 10
                new_max = 1200
                old_min = np.min(mean_data)
                old_max = np.max(mean_data)
                scaled_array = (mean_data - old_min) * (new_max - new_min) / (old_max - old_min) + new_min
                data_to_send = np.array2string(
                    scaled_array.flatten().astype(int),
                    formatter={'int': lambda x: f"{x:04d}"},
                    separator=';'
                )[1:-1]
                data_to_send = data_to_send.replace(' ', '')

                if not self.tcp:
                    self.sock.sendto(
                        bytes(str(str(prediction) + calculated_velocity + ' ' + data_to_send), "utf-8"),
                        (self.ip, self.port)
                    )
                else:
                    self.conn.sendall(
                        str.encode(str(prediction) + calculated_velocity + ' ' + data_to_send + '\n')
                    )

                if self.std_out:
                    print(f":ooooo{int(prediction)} {calculated_velocity} {time.time()}")
    
    
    def _format_data_sample(self, data):
        arr = None
        for feat in data:
            if arr is None:
                arr = data[feat]
            else:
                arr = np.hstack((arr, data[feat]))
        return arr

    def _get_data_helper(self):
        data = np.array(self.raw_data.get_emg())
        if self.filters is not None:
            try:
                data = self.filters.filter(data)
            except:
                pass
        return data