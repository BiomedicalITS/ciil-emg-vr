import socket
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


def worker(config: Config, in_port: int, out_port: int, oclassi: OnlineEMGClassifier):
    save_dir = config.paths.model.replace("model.pth", "")

    logging.basicConfig(
        filename=save_dir + "adaptmanager.log",
        filemode="w",
        encoding="utf-8",
        level=logging.INFO,
    )

    # receive messages from MemoryManager
    in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    in_sock.bind(("localhost", in_port))

    # get an adaptable copy of the model
    # TODO we probably want to Lock the model
    model: EmgCNN = copy.deepcopy(config.model)

    # initialize memory
    memory = Memory()
    memory_id = 0

    # initial time
    start_time = time.perf_counter()

    # variables to save
    adapt_round = 0

    time.sleep(3)
    while time.perf_counter() - start_time < config.game_time:
        try:
            udp_pkt = in_sock.recv(1024).decode()
            if udp_pkt != "WROTE":
                continue
            # new adaptation written, load it in
            # append this data to our memory
            t1 = time.perf_counter()
            memory += Memory().from_file(save_dir, memory_id)
            memory_id += 1
            del_t = time.perf_counter() - t1
            print(f"Loaded {memory_id} memory")
            logging.info(
                f"ADAPTMANAGER: ADDED MEMORIES, \tCURRENT SIZE: {len(memory)}; \tLOAD TIME: {del_t:.2f}s"
            )
            if not len(memory):
                # if we still have no memories (rare edge case)
                logging.info("NO MEMORIES -- SKIPPED TRAINING")
                t1 = time.perf_counter()
                time.sleep(3)
                del_t = time.perf_counter() - t1
                logging.info(
                    f"ADAPTMANAGER: WAITING - round {adapt_round}; \tWAIT TIME: {del_t:.2f}s"
                )
            else:
                if config.adaptation:
                    if config.relabel_method == "LabelSpreading":
                        # TODO I have no idea what this does
                        if time.perf_counter() - start_time > 120:
                            t_ls = time.perf_counter()
                            negative_memory_index = [
                                i == "N" for i in memory.experience_outcome
                            ]
                            labels = memory.experience_targets.argmax(1)
                            labels[negative_memory_index] = -1
                            ls = LabelSpreading(kernel="knn", alpha=0.2, n_neighbors=50)
                            ls.fit(memory.experience_data.numpy(), labels)
                            current_targets = ls.transduction_

                            velocity_metric = torch.mean(memory.experience_data, 1)
                            # two component unsupervised GMM
                            gmm = GaussianMixture(n_components=2).fit(
                                velocity_metric.unsqueeze(1)
                            )
                            gmm_probs = gmm.predict_proba(velocity_metric.unsqueeze(1))
                            gmm_predictions = np.argmax(gmm_probs, 1)
                            lower_cluster = np.argmin(gmm.means_)
                            mask = gmm_predictions == lower_cluster
                            # mask2 = np.max(gmm_probs,1) > 0.95
                            # mask = np.logical_and(mask1, mask2)
                            current_targets[mask] = 2
                            labels = torch.tensor(current_targets, dtype=torch.long)
                            memory.experience_targets = torch.eye(5)[labels]
                            del_t_ls = time.perf_counter() - t_ls
                            logging.info(
                                f"ADAPTMANAGER: LS/GMM - round {adapt_round}; \tLS TIME: {del_t_ls:.2f}s"
                            )

                    t1 = time.perf_counter()

                    model.fit(memory)
                    config.model = model
                    oclassi.classifier.classifier = model

                    del_t = time.perf_counter() - t1

                    logging.info(
                        f"ADAPTMANAGER: ADAPTED - round {adapt_round}; \tADAPT TIME: {del_t:.2f}s"
                    )

                    save_nn(
                        model,
                        config.paths.model.replace(".pth", f"_adapt{adapt_round}.pth"),
                    )

                    print(f"Adapted {adapt_round} times")
                    adapt_round += 1
                else:
                    t1 = time.perf_counter()
                    time.sleep(5)
                    del_t = time.perf_counter() - t1
                    logging.info(
                        f"ADAPTMANAGER: WAITING - round {adapt_round}; \tWAIT TIME: {del_t:.2f}s"
                    )
                    adapt_round += 1

            # tell MemoryManager we are ready for more adaptation data
            in_sock.sendto("WAITING".encode("utf-8"), ("localhost", out_port))
            logging.info("ADAPTMANAGER: WAITING FOR DATA")
            time.sleep(0.5)
        except Exception:
            logging.error("ADAPTMANAGER: " + traceback.format_exc())
    else:
        print("AdaptManager Finished!")
        memory.write(save_dir, 1000)
