import numpy as np
import torch
from torch.nn import functional as F
import socket
from datetime import datetime
from collections import deque

import lightning as L

from libemg.data_handler import OnlineDataHandler

from emager_py import majority_vote

from models import EmgCNN
import utils
import globals as g


def finetune_model(
    model: EmgCNN,
    optimizer: torch.optim.Optimizer,
    samples: torch.Tensor,
    labels: torch.Tensor,
    epochs=10,
):
    model = model.train()
    losses = []
    for _ in range(epochs):
        loss = model.training_step((samples, labels), 0)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        losses.append(loss.item())
    print(losses)
    return model.eval()


def main_loop(model: EmgCNN, odh: OnlineDataHandler, voting_window: int, port: int):
    voter = majority_vote.MajorityVote(voting_window)
    gesture_dict = utils.map_class_to_gestures(g.TRAIN_DATA_DIR)
    optimizer = model.configure_optimizers()

    sample_queue = np.zeros((g.EMG_RUNNING_MEAN_LEN, 1, *g.EMG_DATA_SHAPE))

    # Finetune on batches of 1 s
    ft_n = 0
    ft_data = np.zeros((g.EMG_SAMPLING_RATE, 1, *g.EMG_DATA_SHAPE), dtype=np.float32)
    ft_labels = np.zeros((g.EMG_SAMPLING_RATE,), dtype=np.uint8)

    # batch data in 25 ms then delete the EMG
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as serv:
        serv.bind(("127.0.0.1", port))
        serv.setblocking(False)
        last_shape = (0, 1, *g.EMG_DATA_SHAPE)

        while True:
            odata = odh.get_data().reshape(-1, 1, *g.EMG_DATA_SHAPE)
            if odata.shape == last_shape:
                continue
            n_new = len(odata) - last_shape[0]
            if n_new > len(sample_queue):
                n_new = len(sample_queue)
            last_shape = odata.shape
            odh.raw_data.reset_emg()
            new_data = odata[-n_new:]  # new data

            sample_queue = np.roll(sample_queue, -n_new, axis=0)
            sample_queue[-n_new:] = new_data

            # Process
            processed = utils.process_data(sample_queue)
            samples = torch.from_numpy(processed[-n_new:]).to(g.ACCELERATOR)

            try:
                # Finetune
                servdat = np.frombuffer(serv.recv(2048), dtype=np.uint8)
                # print(f"({datetime.now().time()}): {servdat}")
            except Exception:
                servdat = None

            if servdat is not None:
                free = len(ft_labels) - ft_n
                if free > len(new_data):
                    # Buffers not full, just fill
                    ft_data[ft_n : ft_n + len(new_data)] = processed[-len(new_data) :]
                    ft_labels[ft_n : ft_n + len(new_data)] = np.repeat(
                        servdat, len(new_data)
                    )
                    ft_n += len(new_data)
                else:
                    # Buffers (will be) full, so fill and finetune step
                    ft_data[ft_n : ft_n + free] = processed[-free:]
                    ft_labels[ft_n : ft_n + free] = np.repeat(servdat, free)
                    ft_n = 0
                    model = finetune_model(
                        model,
                        optimizer,
                        torch.from_numpy(ft_data).to(g.ACCELERATOR),
                        torch.from_numpy(ft_labels).to(g.ACCELERATOR),
                    )
                    # fill the first samples
                    ft_data[: len(new_data) - free] = processed[
                        -(len(new_data) - free) :
                    ]
                    ft_labels[: len(new_data) - free] = np.repeat(
                        servdat, len(new_data) - free
                    )
                    ft_n = len(new_data) - free

            pred = model(samples)
            pred = pred.cpu().detach().numpy()
            topk = np.argmax(pred, axis=1)
            voter.extend(topk)
            maj_vote = voter.vote().item(0)
            print(f"{gesture_dict[maj_vote]} ({maj_vote})")


if __name__ == "__main__":
    # Setup the streamers
    utils.setup_streamer()

    # Create data handler and model
    odh = utils.get_online_data_handler(
        g.EMG_SAMPLING_RATE,
        notch_freq=g.EMG_NOTCH_FREQ,
        use_imu=True,
        # max_buffer=g.EMG_SAMPLING_RATE * g.EMG_MAJ_VOTE_MS // 1000,
    )
    model = utils.get_model(True, num_classes=len(g.LIBEMG_GESTURE_IDS))
    model = model.eval()
    # And now main loop
    main_loop(model, odh, g.EMG_MAJ_VOTE_MS * g.EMG_SAMPLING_RATE // 1000, g.UDP_PORT)
