import socket
import time
import datetime


from configs import PSEUDO_LABELS_PORT

if __name__ == "__main__":
    addr = ("127.0.0.1", PSEUDO_LABELS_PORT)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        try:
            s.sendto(b"READY", addr)
            while True:
                print(
                    f"({datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}) Sent fake context."
                )
                message = f"P {time.time()} H1 H2 H3"
                s.sendto(message.encode(), addr)
                time.sleep(0.3)
        finally:
            s.sendto(b"Q", addr)
