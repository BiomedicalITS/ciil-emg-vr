import socket
import time

from nfc_emg import utils
from nfc_emg.schemas import POSE_TO_NAME

if __name__ == "__main__":
    in_addr = ("127.0.0.1", 12347)
    out_addr = ("127.0.0.1", 12350)

    possibilities = "H1 H2"  # set the simulated object possibilities here

    cid_to_name = utils.map_cid_to_ordered_name(
        "data/gestures/", "data/0/bio/adap/train/"
    )
    print(cid_to_name)

    possible_names = [POSE_TO_NAME[p] for p in possibilities.split(" ")]
    print(possible_names)

    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as input_sock:
        input_sock.bind(in_addr)
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as output_sock:
            try:
                output_sock.sendto(b"READY", out_addr)
                while True:
                    classification = input_sock.recv(1024).decode().split(" ")
                    time.sleep(0.01)  # Dumb way to give time to load in new csv data

                    pred = int(classification[0])
                    timestamp = classification[1]

                    if pred == -1:
                        # Rejected
                        continue

                    class_name = cid_to_name[pred]

                    if (
                        class_name not in POSE_TO_NAME.values()
                        or class_name == "No_Motion"
                    ):
                        print(f"{pred} {class_name}")
                        continue

                    if class_name in possible_names:
                        outcome = "P"
                    else:
                        outcome = "N"

                    print(
                        f"{timestamp} Prediction: {pred} ({class_name}),  Possibilities: {possibilities} ({outcome})"
                    )

                    message = f"{outcome} {timestamp} {possibilities}"
                    output_sock.sendto(message.encode(), out_addr)
            except Exception:
                output_sock.sendto(b"STOP", out_addr)
                print("Sent STOP message")
                exit(0)
