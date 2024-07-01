import socket
import time
import random
import datetime


from configs import PSEUDO_LABELS_PORT

if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        while True:
            print(
                f"({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Sent label P"
            )
            s.sendto(b"P H1 H2 H3", ("127.0.0.1", PSEUDO_LABELS_PORT))
            time.sleep(37/1500)
