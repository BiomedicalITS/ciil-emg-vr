import socket
from threading import Lock
import time
import logging
import copy
import csv

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.semi_supervised import LabelSpreading

from libemg.emg_classifier import OnlineEMGClassifier
from libemg.feature_extractor import FeatureExtractor

from nfc_emg.models import save_nn
from nfc_emg import datasets, utils

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

    # Create some initial memory data
    LOAD_INITIAL_DATA = False

    ls_labels = np.ndarray((0,))
    ls_features = np.ndarray(
        (0, len(config.features) * np.prod(config.sensor.emg_shape))
    )

    if LOAD_INITIAL_DATA:
        data_dir = config.paths.get_train()
        base_odh = datasets.get_offline_datahandler(
            data_dir,
            utils.get_cid_from_gid(config.paths.gestures, data_dir, config.gesture_ids),
            utils.get_reps(data_dir),
        )

        base_win, ls_labels = datasets.prepare_data(base_odh, config.sensor)
        ls_features = FeatureExtractor().extract_features(
            config.features, base_win, array=True
        )

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

    logger.info("starting")
    start_time = time.perf_counter()

    while time.perf_counter() - start_time < config.game_time:
        try:
            udp_pkt = manager_sock.recv(1024).decode()
            if udp_pkt == "STOP":
                logger.info("received STOP")
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
                f"loaded memory {memory_id}, size {len(memory)}, load time: {del_t:.2f}s"
            )

            if len(memory) < 2 / 0.05:
                logger.info(f"memory len {len(memory)}. Skipped training")
                time.sleep(1)
            elif config.adaptation:
                if config.relabel_method == "LabelSpreading":
                    t_ls = time.perf_counter()

                    new_p = np.nonzero(np.array(memory.experience_outcome) == "P")
                    new_n = np.nonzero(np.array(memory.experience_outcome) == "N")

                    logging.info(
                        f"Memory len {len(memory)} (P: {len(new_p[0])}, N: {len(new_n[0])})"
                    )

                    # If everything is wrong, don't Label Spread, instead just train with noisy labels.
                    if len(new_p[0]) > 0:
                        # Convert N labels to "-1"
                        new_labels = np.argmax(memory.experience_targets, axis=1)
                        new_labels[new_n] = -1

                        # Create new dataset with P+N
                        adap_labels = np.append(ls_labels, new_labels)
                        adap_features = np.vstack((ls_features, memory.experience_data))

                        # Extend P dataset
                        ls_len = len(ls_labels)
                        ls_labels = np.append(ls_labels, new_labels[new_p])
                        ls_features = np.vstack(
                            (ls_features, memory.experience_data[new_p])
                        )

                        # Fit & predict LS
                        ls = LabelSpreading(kernel="rbf", alpha=0.2, n_neighbors=50)
                        ls.fit(adap_features, adap_labels)

                        # Only retrieve the new adap labels
                        memory.experience_targets = np.eye(len(config.gesture_ids))[
                            ls.transduction_[ls_len:].astype(np.int32)
                        ]

                        # Save transducted
                        memory.write(memory_dir, f"ls_{memory_id}")

                        del_t_ls = time.perf_counter() - t_ls
                        logger.info(f"LabelSpreading time: {del_t_ls:.2f} s")

                adap_data, adap_labels = (
                    memory.experience_data,
                    memory.experience_targets,
                )

                correct = memory.experience_outcome.count("P")
                pre_acc = correct / len(memory.experience_outcome)
                logger.info(f"#{adapt_round+1} pre-acc: {pre_acc*100:.2f}%")

                t1 = time.perf_counter()
                rets = model_to_adapt.fit(adap_data, adap_labels.astype(np.float32))
                del_t = time.perf_counter() - t1

                if rets:
                    adapt_round += 1
                    csv_results.writerow(
                        [adapt_round, len(memory), pre_acc] + list(rets.values())
                    )
                    csv_file.flush()
                    logger.info(f"#{adapt_round} adap time {del_t:.2f} s")

                    new_model = copy.deepcopy(model_to_adapt)

                    # after training, update the model
                    with model_lock:
                        oclassi.classifier.classifier = new_model
                        config.model = new_model

                    save_nn(
                        model_to_adapt,
                        model_path + f"model_{adapt_round}.pth",
                    )
                    memory = Memory()
                else:
                    logger.warning("AM: no adaptation")

            # tell MemoryManager we are ready for more adaptation data
            manager_sock.sendto("WAITING".encode(), mem_manager_addr)
            logger.info("waiting for data")
            time.sleep(5)
        except Exception as e:
            logger.error(f"AM: {e}")
            break
    manager_sock.sendto("STOP".encode(), mem_manager_addr)
    memory.write(memory_dir, 1000)
    logger.info("finished")
