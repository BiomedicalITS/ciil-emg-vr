import socket
import time

if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 5111))
        while True:
            dat = s.recv(2048).decode("utf-8")
            print(dat)
            time.sleep(1)
