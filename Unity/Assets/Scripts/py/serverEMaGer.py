# -*- coding: utf-8 -*-
"""
Created on Tue Nov  7 12:01:04 2023

@author: THLAB40
"""
import socket
from SGT import SGT
import time
import threading
import numpy as np

import libemg
from libemg.data_handler import OnlineDataHandler

import PrepareClassifierEMaGer

def TaskSGT(odh, SGTFlag: bool, client_socket:socket.socket, sgt_list: list):
    while SGTFlag:
        # Receive data from Unity
        dataSGT = client_socket.recv(8192).decode('utf-8')
        print(f"Received data from Unity: {dataSGT}")
        if dataSGT:
            SGTFlag, sgt = process_unity_command(odh, dataSGT, sgt_list[0], SGTFlag)
            sgt_list[0] = sgt  # Update the value in the list

def StartServer(ondh:OnlineDataHandler, tcp_sock_addr: tuple, sgt_list: list[SGT]):
    SGTFlag = True
    try:
        # Wait for Unity game
        server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_socket.bind(tcp_sock_addr)
        server_socket.listen()
        print(f"Server started on {tcp_sock_addr[0]}:{tcp_sock_addr[1]}")
        client_socket, client_address = server_socket.accept()
        print(f"Connection established with {client_address[0]}:{client_address[1]}")

        # Start SGT
        if SGTFlag:
            TaskSGT(ondh, SGTFlag, client_socket, sgt_list)
            print("SGT training over")
            sgt = sgt_list[0]
            print(sgt.inputs_names)
        else:
            sgt = SGT(ondh, 1, 1, 1, "a,b", 'C:\\Users\\GAGAG158\\Documents\\VrGameRFID\\Data')
            sgt.inputs_names = ['Chuck_Grip.png', 'Hand_Close.png', 'Hand_Open.png', 'Index_Extension.png', 'Index_Pinch.png', 'No_Motion.png', 'Wrist_Extension.png', 'Wrist_Flexion.png']

        ondh.stop_listening()

        # Now train + run live classifier
        oc = PrepareClassifierEMaGer.prepare_classifier(sgt.num_reps, sgt.input_count, sgt.output_folder)
        print("Finished preparing classifier. Launching live classification task.")
        classi = PrepareClassifierEMaGer.start_live_classifier(oc, ondh)

        my_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        my_sock.bind(("127.0.0.1", 12350))
        while True:
            data = my_sock.recv(2048)
            print(f"Received data from Unity: {data}")

    except Exception as e:
        print(f"Server error: {e}")      
    finally:
        client_socket.close()
        server_socket.close()

          

def process_unity_command(odh: OnlineDataHandler, data: bytes, sgt: SGT, SGTFlag=True):
    if (data[0] == 'I'): # Initialization
        #SGT(data_handler, num_reps, time_per_reps, time_bet_rep, inputs_names, output_folder)
        parts = data.split(' ')
        num_reps = int(parts[1])
        time_per_rep = int(parts[2])
        time_bet_rep = int(parts[3])
        input_names = parts[4]
        output_folder = parts[5]
        sgt = SGT(odh, num_reps, time_per_rep, time_bet_rep, input_names, output_folder)
        return SGTFlag, sgt
    else:
        SGTFlag = sgt._collect_data(data[0])
        return SGTFlag, sgt


def __main():
    TCP_ADDR = ('127.0.0.1', 12346)  # Python acts as a TCP server at bootup
    p = libemg.streamers.sifibridge_streamer(version="1_1", notch_freq=50)
    odh = libemg.data_handler.OnlineDataHandler()
    odh.start_listening()
    StartServer(odh, TCP_ADDR, [None])
    p.kill()
    
if __name__ == "__main__":
    __main()