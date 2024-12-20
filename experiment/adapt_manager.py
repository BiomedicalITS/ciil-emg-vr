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


def run_adaptation_manager(
    config: Config,
    model_lock: Lock,
    mem_manager_port: int,
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
    with model_lock:
        model_to_adapt = copy.deepcopy(config.model)

    # initialize memory
    memory = Memory()
    memory_id = 0

    # variables to save
    adapt_round = 0

    # comm with MemoryManager
    mem_manager_addr = ("localhost", mem_manager_port)
    manager_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    manager_sock.sendto("WAITING".encode("utf-8"), mem_manager_addr)

    csv_file = open(config.paths.get_results(), "w", newline="")
    csv_results = csv.writer(csv_file)

    logger.info("AM: starting")
    start_time = time.perf_counter()

    while time.perf_counter() - start_time < config.game_time:
        try:
            udp_pkt = manager_sock.recv(1024).decode()
            if udp_pkt == "STOP":
                logger.info("AM: received STOP")
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
                    logger.info(f"AM: LabelSpreading time: {del_t_ls:.2f} s")

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

                preds = model_to_adapt.predict(adap_data)
                acc = accuracy_score(np.argmax(adap_labels, axis=1), preds)
                logger.info(f"AM: #{adapt_round+1} pre-acc: {acc*100:.2f}%")

                t1 = time.perf_counter()
                rets = model_to_adapt.fit(adap_data, adap_labels.astype(np.float32))
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

                    new_model = copy.deepcopy(model_to_adapt)

                    # after training, update the model
                    with model_lock:
                        oclassi.classifier.classifier = new_model
                        config.model = new_model

                    save_nn(
                        model_to_adapt,
                        model_path + f"model_{adapt_round}.pth",
                    )
                else:
                    logger.warning("AM: no adaptation")

            # tell MemoryManager we are ready for more adaptation data
            manager_sock.sendto("WAITING".encode(), mem_manager_addr)
            logger.info("AM: waiting for data")
            time.sleep(5)
        except Exception as e:
            logger.error(f"AM: {e}")
            manager_sock.sendto("STOP".encode(), mem_manager_addr)
            break
    manager_sock.sendto("STOP".encode(), mem_manager_addr)
    memory.write(memory_dir, 1000)
    logger.info("AM: finished")
