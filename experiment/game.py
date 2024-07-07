from multiprocessing import Process
from threading import Thread

import socket

from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier

from nfc_emg.utils import get_online_data_handler
from nfc_emg import models

from config import Config
import memory_manager
import adapt_manager


class Game:
    def __init__(self, config: Config):
        self.classifier_port = 12347
        self.mem_manager_port = 12348
        self.adap_manager_port = 12349
        self.unity_port = 12350

        self.config = config

        self.odh = get_online_data_handler(
            config.sensor,
            imu=False,
            timestamps=True,
            file=True,
            file_path=self.config.paths.live_data,
        )

        # TODO read data from UDP or from file ?

        classi = EMGClassifier()
        classi.add_majority_vote(self.config.sensor.maj_vote_n)

        self.oclassi = OnlineEMGClassifier(
            classi,
            self.config.sensor.window_size,
            self.config.sensor.window_increment,
            self.odh,
            self.config.features,
            port=self.classifier_port,
        )

    def run(self):
        print("Waiting for Unity to send 'READY'...")

        unity_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        unity_sock.bind(("localhost", self.unity_port))
        while True:
            unity_packet = unity_sock.recv(1024).decode()
            if unity_packet == "READY":
                # global_timer = time.perf_counter()
                unity_sock.close()
                break

        Thread(target=lambda: self.oclassi.run(block=True)).start()

        # probably want to start it as thread? Or load model from disk every time?
        Process(
            target=memory_manager.worker,
            daemon=True,
            args=(
                self.config,
                self.adap_manager_port,
                self.unity_port,
                self.mem_manager_port,
            ),
        ).start()

        # Not launched in new process to be able to update the oclassi directly
        adapt_manager.worker(
            self.config,
            self.mem_manager_port,
            self.adap_manager_port,
            self.oclassi,
        )

        # while time.perf_counter() - global_timer < self.config.game_time:
        #     time.sleep(1)

        self.clean_up()
        # because we are running daemon processes they die as main process dies

    def clean_up(self):
        models.save_nn(self.config.paths.model, self.oclassi.classifier)
        self.odh.stop_listening()
        self.oclassi.stop_running()
