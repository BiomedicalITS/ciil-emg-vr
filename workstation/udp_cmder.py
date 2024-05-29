import socket
import time
import random
import json

import globals as g
import utils

if __name__ == "__main__":
    gdict = utils.map_class_to_gestures(g.TRAIN_DATA_DIR)
    no_motion = 0
    directions = ["forward", "backward", "left", "right", "up", "down"]

    for k, v in gdict.items():
        if "no_motion" in v.lower():
            no_motion = k
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        while True:
            # label = random.randint(0, len(g.LIBEMG_GESTURE_IDS) - 1)
            label = no_motion  # no motion
            # direc = random.choice(directions)
            direc = "forward"
            cmd = {"direction": direc, "label": label}
            print(f"Sending {cmd}")
            s.sendto(json.dumps(cmd).encode(), ("192.168.50.39", 5112))
            time.sleep(5)
