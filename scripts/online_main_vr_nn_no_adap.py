import socket
import os
import threading

from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier

from nfc_emg import utils, models
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths

import configs as g

def main_nfcemg_vr():
    sensor = EmgSensor(g.SENSOR, majority_vote_ms=200)
    sensor.start_streamer()

    paths = NfcPaths(f"data/{g.SUBJECT}_{sensor.get_name()}")
    paths.set_trial_number(paths.trial_number - 1)
    paths.gestures = "data/gestures/"
    paths.set_model_name("model_conv1")

    odh = utils.get_online_data_handler(
        sensor.fs, 
        sensor.bandpass_freqs, 
        sensor.notch_freq, 
        False, 
        False if g.SENSOR == EmgSensorType.BioArmband else True,
        timestamps=True, 
        file=True,
        file_path=paths.live_data,
    )

    model = models.load_conv1(paths.model, len(g.FEATURES), sensor.emg_shape)

    classi = EMGClassifier()
    classi.classifier = model
    classi.add_majority_vote(sensor.maj_vote_n)
    classi.add_rejection(0.9)
    oclassi = OnlineEMGClassifier(classi, sensor.window_size, sensor.window_increment, odh, g.FEATURES, port=g.PREDS_PORT)
    
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
    