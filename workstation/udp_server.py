import socket
import time

import globals as g

if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", g.UDP_PORT))
        while True:
            dat = s.recv(2048).decode("utf-8")
            print(dat)
            time.sleep(1)
