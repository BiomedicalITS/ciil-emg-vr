import numpy as np
import torch
import json

from libemg.data_handler import OnlineDataHandler

from emager_py import majority_vote

from models import EmgCNN
import utils
import globals as g


def main_loop(model: EmgCNN, odh: OnlineDataHandler, voting_window: int):
    # TODO watch for fine tuning labels
    # TODO find a better way to watch for new data

    voter = majority_vote.MajorityVote(voting_window)
    with open("data/gestures/gesture_list.json", "r") as f:
        gesture_dict = json.load(f)

    last_shape = None
    while True:
        dat = odh.get_data()
        if len(dat) == 0 or dat.shape == last_shape:
            continue

        last_shape = dat.shape
        dat = dat[last_shape[0] - 1 :]

        # Process + predict
        dat = utils.process_data(dat).reshape(len(dat), 1, 1, 8)
        pred = model(torch.from_numpy(dat.astype(np.float32)).to(model.device))
        pred = pred.cpu().detach().numpy()

        # Majority vote
        voter.append(np.argmax(pred, axis=1))
        maj_vote = voter.vote()
        print(gesture_dict[str(g.LIBEMG_GESTURE_IDS[maj_vote[0]])])


if __name__ == "__main__":
    # Setup the streamers
    utils.setup_streamer()

    # Create data handler and model
    odh = utils.get_online_data_handler(
        g.EMG_SAMPLING_RATE, notch_freq=g.EMG_NOTCH_FREQ, use_imu=g.USE_IMU
    )
    model = utils.get_model()

    # And now main loop
    main_loop(model, odh, g.EMG_MAJ_VOTE_MS * g.EMG_SAMPLING_RATE // 1000)
