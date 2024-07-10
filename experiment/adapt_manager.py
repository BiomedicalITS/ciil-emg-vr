import socket
from threading import Lock
import time
import logging
import traceback
import copy
import csv

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelSpreading

from libemg.emg_classifier import OnlineEMGClassifier

from nfc_emg.models import save_nn

from config import Config
from memory import Memory


def worker(
    config: Config,
    model_lock: Lock,
    in_port: int,
    out_port: int,
    oclassi: OnlineEMGClassifier,
):
    save_dir = config.paths.get_experiment_dir()
    memory_dir = config.paths.get_memory()
    model_path = save_dir + "/models/model_"

    logger = logging.getLogger("adapt_manager")
    fh = logging.FileHandler(save_dir + "adapt_manager.log", mode="w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)

    # "adaptation model copy"
    model_lock.acquire()
    a_model = copy.deepcopy(config.model)
    model_lock.release()

    # initialize memory
    memory = Memory()
    memory_id = 0

    # variables to save
    adapt_round = 0

    print("AdaptManager is starting!")

    # receive messages from MemoryManager
    in_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    in_sock.bind(("localhost", in_port))
    in_sock.sendto("WAITING".encode("utf-8"), ("localhost", out_port))

    # initial time
    start_time = time.perf_counter()
    csv_file = open(config.paths.get_results(), "w", newline="")
    csv_results = csv.writer(csv_file)

    time.sleep(5)
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

            # print(f"Loaded memory #{memory_id} in {del_t:.3f} s")
            logger.info(
                f"ADAPTMANAGER: ADDED MEMORY {memory_id}, CURRENT SIZE: {len(memory)}; LOAD TIME: {del_t:.2f}s"
            )

            if len(memory) < 150:
                logger.info(f"MEMORY LEN {len(memory)} -- SKIPPED TRAINING")
                time.sleep(1)
            else:
                if config.adaptation:
                    if config.relabel_method == "LabelSpreading":
                        if time.perf_counter() - start_time > 120:
                            t_ls = time.perf_counter()
                            n_mem_idx = np.argwhere(memory.experience_outcome == "N")
                            labels = np.argmax(memory.experience_targets, axis=1)
                            labels[n_mem_idx] = -1

                            ls = LabelSpreading(kernel="knn", alpha=0.2, n_neighbors=50)
                            ls.fit(memory.experience_data, labels)

                            current_targets = ls.transduction_
                            memory.experience_targets = np.eye(len(config.gesture_ids))[
                                current_targets
                            ]
                            del_t_ls = time.perf_counter() - t_ls
                            logging.info(
                                f"ADAPTMANAGER: LS - round {adapt_round}; LS TIME: {del_t_ls:.2f}s"
                            )

                    t1 = time.perf_counter()

                    preds = a_model.predict(memory.experience_data)
                    acc = accuracy_score(
                        np.argmax(memory.experience_targets, axis=1), preds
                    )
                    print(
                        f"Round {adapt_round+1} pre-acc: {acc*100:.2f}% ({len(memory)} samples)"
                    )

                    rets = a_model.fit(
                        memory.experience_data,
                        memory.experience_targets.astype(np.float32),
                    )

                    del_t = time.perf_counter() - t1

                    if rets:
                        csv_results.writerow(
                            [adapt_round, len(memory)] + list(rets.values())
                        )
                        csv_file.flush()
                        print(
                            f"Round {adapt_round+1} adap-acc: {rets['acc']*100:.2f}%, adap time {del_t:.2f} s"
                        )

                    new_model = copy.deepcopy(a_model)

                    # after training, update the model
                    model_lock.acquire()
                    oclassi.classifier.classifier = new_model
                    config.model = new_model
                    model_lock.release()

                    logger.info(
                        f"ADAPTMANAGER: ADAPTED - round {adapt_round}; \tADAPT TIME: {del_t:.2f}s"
                    )

                    # save_nn(
                    #     a_model,
                    #     model_path + f"{adapt_round}.pth",
                    # )

                    print(f"Adapted {adapt_round+1} times")

                adapt_round += 1

            # tell MemoryManager we are ready for more adaptation data
            in_sock.sendto("WAITING".encode(), ("localhost", out_port))
            logger.info("ADAPTMANAGER: WAITING FOR DATA")
            time.sleep(5)
        except Exception:
            logging.error("ADAPTMANAGER: " + traceback.format_exc())
            return
    else:
        print("AdaptManager Finished!")
        memory.write(memory_dir, 1000)
