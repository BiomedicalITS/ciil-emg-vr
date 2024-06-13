import socket
import json

import configs as g
from nfc_emg import schemas as sc

if __name__ == "__main__":
    print("*" * 80)
    print("Robot simulator: Make sure to set g.ROBOT_IP to 127.0.0.1")
    print("*" * 80)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        sock = (g.ROBOT_IP, g.ROBOT_PORT)
        s.bind(sock)
        while True:
            cmd = sc.from_dict(json.loads(s.recv(2048)))
            print(cmd)
