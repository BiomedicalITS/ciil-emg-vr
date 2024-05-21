import socket
import time

if __name__ == "__main__":
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        while True:
            print("Sending data....")
            s.sendto(b"Hello, World!", ("127.0.0.1", 5111))
            time.sleep(1)
