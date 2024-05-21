import numpy as np
import torch
import json
import socket

from libemg.data_handler import OnlineDataHandler

from emager_py import majority_vote

from models import EmgCNN
import utils
import globals as g


def main_loop(
    model: EmgCNN, odh: OnlineDataHandler, voting_window: int, port: int = 5111
):
    # TODO watch for fine tuning labels
    # TODO find a better way to watch for new data
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as serv:
        serv.bind(("127.0.0.1", port))

        voter = majority_vote.MajorityVote(voting_window)
        with open("data/gestures/gesture_list.json", "r") as f:
            gesture_dict = json.load(f)

        queue_len = g.EMG_RUNNING_MEAN_LEN
        sample_queue = np.zeros((queue_len, 1, *g.EMG_DATA_SHAPE))

        last_shape = None
        while True:
            dat = odh.get_data()
            if len(dat) == 0 or dat.shape == last_shape:
                continue

            try:
                # TODO THIS!
                servdat = serv.recv(2048).decode("utf-8")
                print(servdat)
            except socket.timeout:
                pass

            last_shape = dat.shape

            dat = dat[last_shape[0] - 1 :]  # new data
            dat = dat.reshape((-1, 1, *g.EMG_DATA_SHAPE))

            sample_queue = np.concatenate((sample_queue, dat), axis=0)[len(dat) :]

            # Process
            processed = utils.process_data(sample_queue)

            # Predict
            pred = model(torch.from_numpy(processed[-len(dat) :]).to(g.ACCELERATOR))
            pred = pred.cpu().detach().numpy()

            # Majority vote
            topk = np.argmax(pred, axis=1)
            voter.extend(topk)
            maj_vote = voter.vote().item(0)
            # TODO libemg does not map in order correctly screen guided training to labels
            print(gesture_dict[str(g.LIBEMG_GESTURE_IDS[maj_vote])])


if __name__ == "__main__":
    # Setup the streamers
    utils.setup_streamer()

    # Create data handler and model
    odh = utils.get_online_data_handler(
        g.EMG_SAMPLING_RATE, notch_freq=g.EMG_NOTCH_FREQ, use_imu=g.USE_IMU
    )
    model = utils.get_model(False, num_classes=len(g.LIBEMG_GESTURE_IDS))
    model = model.eval()
    # And now main loop
    main_loop(model, odh, g.EMG_MAJ_VOTE_MS * g.EMG_SAMPLING_RATE // 1000)
