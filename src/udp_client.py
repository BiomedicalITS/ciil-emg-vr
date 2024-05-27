import socket
import time
import random

import globals as g
import utils

if __name__ == "__main__":
    gdict = utils.map_class_to_gestures(g.TRAIN_DATA_DIR)
    no_motion = 0
    for k, v in gdict.items():
        if "no_motion" in v.lower():
            no_motion = k
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        while True:
            # label = random.randint(0, len(g.LIBEMG_GESTURE_IDS) - 1)
            label = no_motion  # no motion
            print(f"Sending {label}")
            s.sendto(bytes([label]), ("127.0.0.1", g.UDP_PORT))
            time.sleep(0.025)
