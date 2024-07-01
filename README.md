# Self-supervised EMG gesture recognition

This project investigates the implementation of a self-supervised EMG gesture recognition system with the help of contextual information.

## Set up

Clone this repository, then:

- `python3 -m venv .venv`
- `source .venv/bin/activate`
- `python3 -m pip install -e ./`

## Using nfc-emg

Leveraging LibEMG, we reuse the paradigm of a hardware-agnostic _OnlineDataHandler_ reading in live data from a UDP _streamer_.

These are the main scripts of interest in `workstation/`:

- `offline_main.py` is executed to train the initial model. It is also used to finetune the model in an offline manner. Screen Guided Training is used for that.
- `online_main.py` is the live experiment script. It starts by calibrating the IMU. Then, the `run()` function infinitely loops. It uses the IMU for 3D cartesian position. EMG is used for wrist and grip detection. The commands are sent to a UDP port. An input UDP socket is also opened to receive self-supervised labels.
- `globals.py` defines system-wide parameters. This file is the main way of configuring the system.
- `schemas.py` is used to interface with an external robot control system. It should be imported on the client-side to ensure compatibility.

## General overview

### NFC

As a first step, NFC tags are programmed with object IDs and placed on the corresponding objects. Upon scanning a tag, for example by having an NFC reader embedded inside of a prosthetic, pseudo-labels are generated, allowing the model to be retrained on the fly.

In this scenario, a UR5 robotic arm is used for live demonstrations. Its 3D position is derived from an IMU, while the finer wrist and finger movements are extracted from the self-supervised model.

### VR

In VR, virtual objects can be freely placed in a simulated environment, eliminating the need to program and place NFC tags on physical objects. Also, arm and hand tracking is offered by the headset. This alternative setup helps to replicate experiments more effectively and follows in the current trends in using VR to ease the training for EMG prosthetics.

## Ideas

- execute a python script via ssh upon scanning an NFC tag with cellphone
- create object classes (shapes?) which are associated with metadata such as grip types
- program the NFC tags with the classes and stick them on objects
- when an NFC tag is scanned, label the next X s of data
- even better: label until significant IMU data is detected, meaning the arm is
- for each labelled "object type", use the IMU data to guess the correct label: eg a cup from the top is a "hand open", from the side "chuck grip (pinch)" (handle) or "power grip"

## Resources

- [VR project](https://github.com/ThomasLabbe01/VrGameRFID)
- [LibEMG doc](https://libemg.github.io/libemg/#)
- [LibEMG gesture library](https://github.com/libemg/LibEMGGestures)

## Random notes

from metadata.json
0 = hand close
1 = hand open
2 = pinch
3 = no motion
4 = extension
5 = flexion

in globals
1 = no motion
2 = hand close
3 = hand open
4 = flexion
5 = extension
26 = pinch
