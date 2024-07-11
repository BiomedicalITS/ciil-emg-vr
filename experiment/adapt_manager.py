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
from sklearn.utils import resample

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
    """
    The AdaptManager is responsible for doing the live adaptation of the model.

    To do so, it waits until MemoryManager writes a "Memory" to disk, then loads it in.

    If the Memory is big enough, it does an adaptation pass on the model, and then saves the model.

    Finally, the OnlineEMGClassifier and the config are updated with the new model.
    """

    save_dir = config.paths.get_experiment_dir()
    memory_dir = config.paths.get_memory()
    model_path = config.paths.get_models()

    logger = logging.getLogger("adapt_manager")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(save_dir + "adapt_manager.log", mode="w")
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    fs = logging.StreamHandler()
    fs.setLevel(logging.INFO)
    logger.addHandler(fs)

    # "adaptation model copy"
    model_lock.acquire()
    a_model = copy.deepcopy(config.model)
    model_lock.release()

    # initialize memory
    memory = Memory()
    memory_id = 0

    # variables to save
    adapt_round = 0

    logger.info("AM: starting")

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
                f"AM: loaded memory {memory_id}, size {len(memory)}, load time: {del_t:.2f}s"
            )

            if len(memory) < 150:
                logger.info(f"AM: memory len {len(memory)}. Skipped training")
                time.sleep(1)
            elif config.adaptation:
                if config.relabel_method == "LabelSpreading":
                    t_ls = time.perf_counter()
                    labels = np.argmax(memory.experience_targets, axis=1)
                    labels[np.argwhere(memory.experience_outcome == "N")] = -1

                    ls = LabelSpreading(kernel="knn", alpha=0.2, n_neighbors=50)
                    ls.fit(memory.experience_data, labels)

                    current_targets = ls.transduction_
                    memory.experience_targets = np.eye(len(config.gesture_ids))[
                        current_targets
                    ]
                    del_t_ls = time.perf_counter() - t_ls
                    logger.info(f"AM: LS TIME: {del_t_ls:.2f} s")

                adap_data, adap_labels = (
                    memory.experience_data,
                    memory.experience_targets,
                )

                # adap_data, adap_labels = resample(
                #     memory.experience_data,
                #     memory.experience_targets,
                #     replace=False,
                #     n_samples=1000,
                # )

                preds = a_model.predict(adap_data)
                acc = accuracy_score(np.argmax(adap_labels, axis=1), preds)
                logger.info(f"AM: #{adapt_round+1} pre-acc: {acc*100:.2f}%")

                t1 = time.perf_counter()
                rets = a_model.fit(adap_data, adap_labels.astype(np.float32))
                del_t = time.perf_counter() - t1

                if rets:
                    adapt_round += 1
                    csv_results.writerow(
                        [adapt_round, len(memory)] + list(rets.values())
                    )
                    csv_file.flush()
                    logger.info(
                        f"AM: #{adapt_round} adap-acc: {rets['acc']*100:.2f}%, adap time {del_t:.2f} s"
                    )

                    new_model = copy.deepcopy(a_model)

                    # after training, update the model
                    model_lock.acquire()
                    oclassi.classifier.classifier = new_model
                    config.model = new_model
                    model_lock.release()

                    save_nn(
                        a_model,
                        model_path + f"model_{adapt_round}.pth",
                    )

            # tell MemoryManager we are ready for more adaptation data
            in_sock.sendto("WAITING".encode(), ("localhost", out_port))
            logger.info("AM: waiting for data")
            time.sleep(5)
        except Exception:
            logger.error("AM: " + traceback.format_exc())
            return
    memory.write(memory_dir, 1000)
    logger.info("AM: finished")
