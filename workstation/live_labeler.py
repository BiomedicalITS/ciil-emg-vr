import sys
import socket
import datetime

from globals import PEUDO_LABELS_PORT

if __name__ == "__main__":
    label = 3
    if len(sys.argv) > 1:
        label = int(sys.argv[1])
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        print(
            f"({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Sent label {label}"
        )
        s.sendto(bytes([label]), ("127.0.0.1", PEUDO_LABELS_PORT))
