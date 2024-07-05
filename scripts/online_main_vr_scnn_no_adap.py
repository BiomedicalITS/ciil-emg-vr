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
    SUBJECT = "vr"
    SENSOR = EmgSensorType.BioArmband

    sensor = EmgSensor(SENSOR, window_inc_ms=5)
    sensor.start_streamer()

    paths = NfcPaths(f"data/{SUBJECT}_{sensor.get_name()}")
    paths.set_trial_number(paths.trial_number - 1)
    paths.gestures = "data/gestures/"
    paths.set_model_name("model_scnn")

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

    model = EmgSCNNWrapper.load_from_disk(paths.model, sensor.emg_shape, "cuda")
    model.model.eval() # feature extractor always in eval

    classi = EMGClassifier()
    classi.classifier = model
    classi.add_majority_vote(sensor.maj_vote_n)
    classi.add_rejection(0.8)
    oclassi = OnlineEMGClassifier(classi, sensor.window_size, sensor.window_increment, odh, ["MAV"], port=g.PREDS_PORT)
    
    # Delete old live  data
    context = []
    try:
        for f in os.listdir(f"{paths.base}/{paths.trial_number}"):
            if not f.startswith("live_"):
                continue
            os.remove(f"{paths.base}/{paths.trial_number}/{f}")
    except Exception:
        pass

    try:
        print("Starting online classification")
        threading.Thread(target=lambda: oclassi.run(), daemon=True).start()

        # read the lines as they come in
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as udp_sock:
            udp_sock.bind(("127.0.0.1", g.PSEUDO_LABELS_PORT))
            while True:
                udp_ctx = udp_sock.recv(2048).decode()
                print(udp_ctx)
                context.append(" ".join(udp_ctx))
                if udp_ctx[0] == "Q":
                    # sent when Unity app is shut down
                    break
    finally:
        if len(context) > 0:
            with open(paths.live_data + "context.txt", "w") as f:
                f.write('\n'.join(context))
        sensor.stop_streamer()
        odh.stop_listening()
        
if __name__ == "__main__":
    main_nfcemg_vr()
    