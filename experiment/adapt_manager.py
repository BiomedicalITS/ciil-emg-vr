import socket
from threading import Lock
import time
import logging
import traceback
import copy

import numpy as np
import torch
from sklearn.semi_supervised import LabelSpreading
from sklearn.mixture import GaussianMixture

from libemg.emg_classifier import OnlineEMGClassifier

from nfc_emg.models import EmgCNN, save_nn

from config import Config
from memory import Memory


def worker(
    config: Config,
    model_lock: Lock,
    in_port: int,
    out_port: int,
    oclassi: OnlineEMGClassifier,
):
    save_dir = config.paths.base + f"/{config.paths.trial_number}/"
    memory_dir = save_dir + "memory/"

    logger = logging.getLogger("adapt_manager")
    fh = logging.FileHandler(save_dir + "adapt_manager.log", mode="w", encoding="utf-8")
    fh.setLevel(logging.INFO)
    fs = logging.StreamHandler()
    fs.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.addHandler(fs)

    # receive messages from MemoryManager
    in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    in_sock.bind(("localhost", in_port))

    # "adaptation model"
    model_lock.acquire()
    a_model = copy.deepcopy(config.model)
    model_lock.release()

    # initialize memory
    memory = Memory()
    memory_id = 0

    # initial time
    start_time = time.perf_counter()

    # variables to save
    adapt_round = 0
    in_sock.sendto("WAITING".encode("utf-8"), ("localhost", out_port))

    time.sleep(3)
    while time.perf_counter() - start_time < config.game_time:
        try:
            udp_pkt = in_sock.recv(1024).decode()
            if udp_pkt == "STOP":
                break
            elif udp_pkt != "WROTE":
                continue

            # new adaptation written, load it in
            # append this data to our memory

            t1 = time.perf_counter()
            memory += Memory().from_file(memory_dir, memory_id)
            memory_id += 1
            del_t = time.perf_counter() - t1

            print(f"Loaded {memory_id} memory")
            logger.info(
                f"ADAPTMANAGER: ADDED MEMORIES, \tCURRENT SIZE: {len(memory)}; \tLOAD TIME: {del_t:.2f}s"
            )

            if len(memory) < 64:
                logger.info(f"MEMORY LEN {len(memory)} -- SKIPPED TRAINING")
                time.sleep(3)
            else:
                if config.adaptation:
                    # if config.relabel_method == "LabelSpreading":
                    #     # TODO I have no idea what this does
                    #     if time.perf_counter() - start_time > 120:
                    #         t_ls = time.perf_counter()
                    #         negative_memory_index = [
                    #             i == "N" for i in memory.experience_outcome
                    #         ]
                    #         labels = memory.experience_targets.argmax(1)
                    #         labels[negative_memory_index] = -1
                    #         ls = LabelSpreading(kernel="knn", alpha=0.2, n_neighbors=50)
                    #         ls.fit(memory.experience_data.numpy(), labels)
                    #         current_targets = ls.transduction_

                    #         velocity_metric = torch.mean(memory.experience_data, 1)
                    #         # two component unsupervised GMM
                    #         gmm = GaussianMixture(n_components=2).fit(
                    #             velocity_metric.unsqueeze(1)
                    #         )
                    #         gmm_probs = gmm.predict_proba(velocity_metric.unsqueeze(1))
                    #         gmm_predictions = np.argmax(gmm_probs, 1)
                    #         lower_cluster = np.argmin(gmm.means_)
                    #         mask = gmm_predictions == lower_cluster
                    #         # mask2 = np.max(gmm_probs,1) > 0.95
                    #         # mask = np.logical_and(mask1, mask2)
                    #         current_targets[mask] = 2
                    #         labels = torch.tensor(current_targets, dtype=torch.long)
                    #         memory.experience_targets = torch.eye(5)[labels]
                    #         del_t_ls = time.perf_counter() - t_ls
                    #         logging.info(
                    #             f"ADAPTMANAGER: LS/GMM - round {adapt_round}; \tLS TIME: {del_t_ls:.2f}s"
                    #         )

                    t1 = time.perf_counter()
                    print("=" * 100)
                    print(len(memory))
                    a_model.fit(memory.experience_data, memory.experience_targets)
                    print("=" * 100)

                    new_model = copy.deepcopy(a_model)

                    # after training, update the model
                    model_lock.acquire()
                    oclassi.classifier.classifier = new_model
                    model_lock.release()

                    del_t = time.perf_counter() - t1

                    logger.info(
                        f"ADAPTMANAGER: ADAPTED - round {adapt_round}; \tADAPT TIME: {del_t:.2f}s"
                    )

                    save_nn(
                        a_model,
                        config.paths.model.replace(".pth", f"_adapt{adapt_round}.pth"),
                    )

                    print(f"Adapted {adapt_round} times")
                    adapt_round += 1
                else:
                    time.sleep(5)
                    adapt_round += 1

            # tell MemoryManager we are ready for more adaptation data
            in_sock.sendto("WAITING".encode(), ("localhost", out_port))
            logger.info("ADAPTMANAGER: WAITING FOR DATA")
            time.sleep(0.5)
        except Exception as e:
            in_sock.sendto("ERROR".encode(), ("localhost", out_port))
            logging.error("ADAPTMANAGER: " + traceback.format_exc())
            return
    else:
        print("AdaptManager Finished!")
        memory.write(memory_dir, 1000)
