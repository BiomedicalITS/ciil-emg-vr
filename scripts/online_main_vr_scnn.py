import socket
import os
import shutil
import copy
import threading
from datetime import datetime
import time
import threading
from collections import deque 

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.utils import shuffle

import numpy as np

from libemg.data_handler import OnlineDataHandler
from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier
from libemg.data_handler import get_windows
from libemg.feature_extractor import FeatureExtractor

from nfc_emg import utils, models, datasets
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths
from nfc_emg.models import EmgSCNN, EmgSCNNWrapper, main_train_scnn
from nfc_emg.schemas import POSE_TO_NAME

import configs as g
from sgt_vr import SGT

def online_classification(oclassi: OnlineEMGClassifier):
    oclassi.run()

def wait_for_unity(tcp_port: int):
    # Wait for Unity game
    print("Waiting for Unity Game to start...")
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind(("127.0.0.1", tcp_port))
    server_socket.listen()
    print(f"Server started on port {tcp_port}")
    client_socket, client_address = server_socket.accept()
    print(f"Connection established with {client_address[0]}:{client_address[1]}")
    return server_socket, client_socket

def do_vr_sgt(odh, client_socket:socket.socket) -> SGT:
    sgt = None
    sgt_flag = True
    client_socket.settimeout(0.05)
    while sgt_flag:
        try:
            dataSGT = client_socket.recv(8192).decode('utf-8')
        except TimeoutError:
            continue
        if dataSGT == "":
            raise WindowsError("TCP socket received empty data, assuing it'S dead!")
        
        print(f"Received data from Unity: {dataSGT}")
        if dataSGT:
            sgt_flag, sgt = process_unity_command(odh, dataSGT, sgt)
    return sgt     

def process_unity_command(odh: OnlineDataHandler, data: bytes, sgt: SGT):
    sgt_flag = True
    if (data[0] == 'I'): # Initialization
        #SGT(data_handler, num_reps, time_per_reps, time_bet_rep, inputs_names, output_folder)
        parts = data.split(' ')
        num_reps = int(parts[1])
        time_per_rep = int(parts[2])
        time_bet_rep = int(parts[3])
        input_names = parts[4]
        output_folder = parts[5]
        sgt = SGT(odh, num_reps, time_per_rep, time_bet_rep, input_names, output_folder)
        return sgt_flag, sgt
    else:
        sgt_flag = sgt._collect_data(data[0])
        return sgt_flag, sgt

def main_nfcemg_vr():
    SENSOR = EmgSensorType.BioArmband
    GESTURE_IDS = g.FUNCTIONAL_SET

    SAMPLE_DATA = True # Train model from freshly sampled data
    DEBUG = False # used to skip SGT and TCP

    SAMPLE_DATA = False 
    # DEBUG = True

    
    sensor = EmgSensor(SENSOR)

    paths = NfcPaths(f"data/vr_{sensor.get_name()}")
    if not SAMPLE_DATA or DEBUG:
        paths.set_trial_number(paths.trial_number - 1)
    paths.gestures = "data/gestures/"
    paths.set_model_name("model_scnn")

    # Delete old live context
    for f in os.listdir(f"{paths.base}/{paths.trial_number}"):
        if not f.startswith("live_"):
            continue
        os.remove(f"{paths.base}/{paths.trial_number}/{f}")

    try:
        print(utils.get_name_from_gid(paths.gestures, paths.train, GESTURE_IDS))
    except Exception:
        pass

    sensor.start_streamer()
    odh = utils.get_online_data_handler(
        sensor.fs, 
        sensor.bandpass_freqs, 
        sensor.notch_freq, 
        False, 
        False if SENSOR == EmgSensorType.BioArmband else True,
    )

    fe = FeatureExtractor()
    model = EmgSCNNWrapper(EmgSCNN(sensor.emg_shape), LinearDiscriminantAnalysis())
    
    training_passes = 1
    ft_data = np.zeros((0, np.prod(sensor.emg_shape)))
    ft_labels = np.zeros((0,), dtype=np.int64)
    context = []

    server_socket, client_socket = socket.socket(), socket.socket()
    try:
        # wait for Unity to be ready
        if DEBUG:
            t = threading.Thread(target=lambda: wait_for_unity(g.WAIT_TCP_PORT))
            t.start()
        else:
            server_socket, client_socket = wait_for_unity(g.WAIT_TCP_PORT)
        
        # Do SGT and copy data from remote project
        if SAMPLE_DATA:
            if not DEBUG:
                sgt = do_vr_sgt(odh, client_socket)
                assert(len(sgt.inputs_names) == len(GESTURE_IDS))
            else:
                sgt = SGT(odh, 5, 3, 2, ",", "C:/Users/GAGAG158/Documents/VrGameRFID/Data")
            try:
                shutil.rmtree(paths.train)
            except FileNotFoundError:
                pass
            os.makedirs(paths.train, exist_ok=True)
            # Copy SGT files to respect paths
            for f in os.listdir(sgt.output_folder):
                dest_name = f
                if f.endswith(".csv"):
                    dest_name = dest_name.replace(".csv", "_EMG.csv")
                shutil.copy(f"{sgt.output_folder}/{f}", paths.train + dest_name)
            model = main_train_scnn(sensor, paths.train, False, GESTURE_IDS, paths.gestures, LinearDiscriminantAnalysis())
            model.save_to_disk(paths.model)
        else:
            model = EmgSCNNWrapper.load_from_disk(paths.model, sensor.emg_shape, "cuda")

        model.model.eval() # feature extractor always in eval

        # Preload some finetuning data
        offdh = datasets.get_offline_datahandler(paths.train, GESTURE_IDS, utils.get_reps(paths.train)[-1:])
        tdata, tlabels = datasets.prepare_data(offdh, sensor)
        tdata = fe.getMAVfeat(windows)
        ft_data = np.vstack((ft_data, tdata))
        ft_labels = np.vstack((ft_labels, tlabels))

        # create ODH with data recording
        odh.stop_listening() 
        odh = utils.get_online_data_handler(
            sensor.fs, 
            sensor.bandpass_freqs, 
            sensor.notch_freq, 
            False, 
            False if SENSOR == EmgSensorType.BioArmband else True,
            timestamps=True, 
            file=True,
            file_path=paths.live_data,
        )
        
        classi = EMGClassifier()
        classi.classifier = model
        classi.add_majority_vote(sensor.maj_vote_n)
        oclassi = OnlineEMGClassifier(classi, sensor.window_size, sensor.window_increment, odh, ["MAV"], port=g.PREDS_PORT, std_out=False)
        threading.Thread(target=online_classification, args=(oclassi,)).start()

        w_model = copy.deepcopy(model)
        name_to_cid_map = utils.reverse_dict(utils.map_cid_to_ordered_name(paths.gestures, paths.train, GESTURE_IDS))
        line_q = deque(maxlen=sensor.fs)
        last_timestamp = 0
        timestamp = 0

        # Wait for data to start being written
        while "live_EMG.csv" not in os.listdir(f"{paths.base}/{paths.trial_number}"):
            continue
        
        # read the lines as they come in
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            udp_sock.bind(("127.0.0.1", g.PSEUDO_LABELS_PORT))
            with open(paths.live_data + "EMG.csv", "r") as csv:
                while True:
                    udp_ctx = udp_sock.recv(2048).decode().split() # sent every Unity frame when near object
                    # TODO handle if nothing is received for a while: reset the last_timestamp? maybe the deque is enough?
                    last_timestamp = timestamp
                    timestamp = float(udp_ctx[1])
                    if udp_ctx[0] == "N":
                        # Not within context so do nothing
                        continue
                    ctx_labels = [name_to_cid_map[POSE_TO_NAME[p]] for p in udp_ctx[2:]]

                    # Read all unread lines
                    newlines = csv.readlines()
                    if len(newlines) == 0:
                        continue
                    newlines = np.fromstring("".join(newlines).replace("\n", ","), sep=",").reshape(-1,9)
                    print(f"Number of new lines: {len(newlines)}")

                    line_q.extend(newlines)
                    data = np.array(line_q)

                    stamps = data[:, 0] # all timestamps from line queue
                    valid_samples = np.squeeze(np.argwhere(np.logical_and(stamps > last_timestamp, stamps <= timestamp))) # find new data but still older than context
                    if len(valid_samples) < sensor.window_size:
                        # not enough new samples
                        continue
                    data = data[valid_samples, 1:]

                    # find which predictions are within-context, meaning good examples
                    new_windows = get_windows(data, sensor.window_size, sensor.window_increment)
                    new_data = fe.getMAVfeat(new_windows)
                    new_labels = w_model.predict(new_data) 
                    
                    ft_data = np.vstack((ft_data, data))
                    context.append(" ".join(udp_ctx))
                    
                    print(f"{data.shape} {ft_data.shape} ({ft_data.nbytes/1024:.0f} kB)")

                    if len(ft_data) > 3*training_passes*sensor.fs:
                        windows = get_windows(ft_data, sensor.window_size, sensor.window_increment)
                        features = fe.getMAVfeat(windows).astype(np.float32)
                        print(features.shape)
                        labels = w_model.predict(features)
                        features, labels = shuffle(features, labels)
                        w_model.fit(features, labels)
                        
                        # Now update the system and reset data buffers
                        model = copy.deepcopy(w_model)
                        oclassi.classifier.classifier = model

                        print(f"Finished train pass #{training_passes}")
                        training_passes += 1
    finally:
        if len(context) > 0:
            with open(paths.live_data + "context.txt", "w") as f:
                f.write('\n'.join(context))
        server_socket.close()
        client_socket.close()
        
if __name__ == "__main__":
    main_nfcemg_vr()
    