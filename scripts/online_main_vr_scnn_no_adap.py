import socket
import os
import threading

from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier

from nfc_emg import utils
from nfc_emg.sensors import EmgSensor, EmgSensorType
from nfc_emg.paths import NfcPaths
from nfc_emg.models import EmgSCNNWrapper

import configs as g

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
    