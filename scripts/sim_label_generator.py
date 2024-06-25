import socket
import time
import random
import datetime


from configs import PSEUDO_LABELS_PORT

if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        while True:
            label = random.choice([2, 3])
            label = 0
            print(
                f"({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Sent label {label}"
            )
            s.sendto(bytes([label]), ("127.0.0.1", PSEUDO_LABELS_PORT))
            time.sleep(0.5)
