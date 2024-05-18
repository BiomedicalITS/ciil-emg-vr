from multiprocessing import Process
import subprocess as sp
import time


def start_sp():
    # stays alive but doesn't leak with -i
    # stays alive and leaks with -o
    # dies sometimes with -i and -o
    p = sp.Popen(
        # ["sifi_bridge"],
        # ["sifi_bridge", "-i", "127.0.0.1:1234"],
        ["sifi_bridge", "-o", "127.0.0.1:1235"],
        # ["sifi_bridge", "-i", "127.0.0.1:1234", "-o", "127.0.0.1:1235"],
        stdin=sp.PIPE,
        stdout=sp.PIPE,
    )
    while True:
        p.stdin.write(b"print('Hello')\n")
        print(p.stdout.readline().decode())
        time.sleep(1)


if __name__ == "__main__":
    p = Process(target=start_sp, daemon=True)
    p.start()
    # time.sleep(8)
    input()
