from multiprocessing import Process
from threading import Lock, Thread
import os

import socket

from libemg.emg_classifier import EMGClassifier, OnlineEMGClassifier

from nfc_emg.utils import get_online_data_handler
from nfc_emg import models

from config import Config
import memory_manager
import adapt_manager
from super_classi import run_classifier


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

        classi = EMGClassifier()
        classi.classifier = config.model.to("cuda").eval()
        classi.add_majority_vote(self.config.sensor.maj_vote_n)

        self.oclassi = OnlineEMGClassifier(
            classi,
            self.config.sensor.window_size,
            self.config.sensor.window_increment,
            self.odh,
            self.config.features,
            port=self.classifier_port,
        )
        self.model_lock = Lock()

        # Delete old data if applicable
        for f in os.listdir(config.paths.base + f"/{config.paths.trial_number}/"):
            if not f.startswith("live_"):
                continue
            os.remove(f"{config.paths.base + f"/{config.paths.trial_number}/"}/{f}")

    def run(self):
        print("Waiting for Unity to send 'READY'...")

        # before running streamer, oclassi, memoryManager and adaptManager, consumes ~20% of CPU

        # unity_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        # unity_sock.bind(("localhost", self.unity_port))
        # while True:
        #     unity_packet = unity_sock.recv(1024).decode()
        #     if unity_packet == "READY":
        #         # global_timer = time.perf_counter()
        #         unity_sock.close()
        #         break

        print("Starting the Python-side of the game stage!")

        self.config.sensor.start_streamer()

        Thread(
            target=run_classifier,
            args=(self.oclassi, self.config.paths.live_data + "preds.csv", self.model_lock),
        ).start()

        Thread(
            target=memory_manager.worker,
            args=(
                self.config,
                self.model_lock,
                self.adap_manager_port,
                self.unity_port,
                self.mem_manager_port,
            ),
        ).start()

        adapt_manager.worker(
            self.config,
            self.model_lock,
            self.mem_manager_port,
            self.adap_manager_port,
            self.oclassi,
        )

        # while time.perf_counter() - global_timer < self.config.game_time:
        #     time.sleep(1)

        self.clean_up()
        # because we are running daemon processes they die as main process dies

    def clean_up(self):
        self.config.model = self.oclassi.classifier.classifier
        models.save_nn(self.config.model, self.config.paths.model)
        self.odh.stop_listening()
        self.oclassi.stop_running()
