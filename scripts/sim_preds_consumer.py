import socket
import json

from nfc_emg import schemas as sc

import configs as g

if __name__ == "__main__":
    print("*" * 80)
    print("Preds Consumer Simulator: Make sure to set g.PREDS_IP to 127.0.0.1")
    print("*" * 80)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        sock = (g.PREDS_IP, 5678)
        s.bind(sock)
        while True:
            # cmd = sc.from_dict(json.loads(s.recv(2048)))
            cmd = s.recv(2048).decode()
            print(cmd)
