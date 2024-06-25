import socket
import os
import shutil

import numpy as np
import torch

from libemg.data_handler import OfflineDataHandler, OnlineDataHandler
from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier
from libemg.feature_extractor import FeatureExtractor

from nfc_emg import utils
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths
from nfc_emg.models import EmgMLP, main_train_nn, main_test_nn, get_model_cnn

import configs as g
from custom_sgt import SGT

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

    SAMPLE_DATA = False
    # SAMPLE_DATA = True
    DEBUG = False

    sensor = EmgSensor(SENSOR)
    paths = NfcPaths(f"data/vr_{sensor.get_name()}", 0)
    paths.set_model_name("model_mlp")
    paths.gestures = "data/gestures/"

    sensor.start_streamer()
    odh = utils.get_online_data_handler(sensor.fs, sensor.bandpass_freqs, imu=False, attach_filters=False)

    fe = FeatureExtractor()
    fg = fe.get_feature_groups()["HTD"]
    model = EmgMLP(len(fg) * np.prod(sensor.emg_shape), len(GESTURE_IDS))
    
    try:
        server_socket, client_socket = wait_for_unity(g.WAIT_TCP_PORT)

        # Do SGT and copy data from remote project
        if SAMPLE_DATA:
            if not DEBUG:
                sgt = do_vr_sgt(odh, client_socket)
                assert(len(sgt.inputs_names) == len(GESTURE_IDS))
            else:
                sgt = SGT(odh, 5, 3, 2, ",", "C:/Users/GAGAG158/Documents/VrGameRFID/Data")
            shutil.rmtree(paths.train)
            os.makedirs(paths.train, exist_ok=True)
            # Copy SGT files to respect paths
            for f in os.listdir(sgt.output_folder):
                dest_name = f
                if f.endswith(".csv"):
                    dest_name = dest_name.replace(".csv", "_EMG.csv")
                shutil.copy(f"{sgt.output_folder}/{f}", paths.train + dest_name)
            
        model = main_train_nn(model, sensor, False, fg, GESTURE_IDS, paths.gestures, paths.train, paths.model)

        classi = EMGClassifier()
        classi.add_majority_vote(sensor.maj_vote_n)
        classi.classifier = model.eval()

        oclassi = OnlineEMGClassifier(classi, sensor.window_size, sensor.window_increment, odh, fg, port=g.PREDS_PORT, std_out=True)
        oclassi.run(block=False)

        udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        udp_sock.bind(("127.0.0.1", g.PSEUDO_LABELS_PORT))
        while True:
            data = udp_sock.recv(2048).decode()
            print(f"Received data from Unity: {data}")    
    finally:
        server_socket.close()
        client_socket.close()
        udp_sock.close()
        
if __name__ == "__main__":
    main_nfcemg_vr()
    