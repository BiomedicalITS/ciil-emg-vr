# Self-supervised EMG gesture recognition

This project investigates the implementation of a self-supervised EMG gesture recognition system with the help of contextual information.

## General overview

### NFC

As a first step, NFC tags are programmed with object IDs and placed on the corresponding objects. Upon scanning a tag, for example by having an NFC reader embedded inside of a prosthetic, pseudo-labels are generated, allowing the model to be retrained on the fly.

In this scenario, a UR5 robotic arm is used for live demonstrations. Its 3D position is derived from an IMU, while the finer wrist and finger movements are extracted from the self-supervised model.

### VR

In VR, virtual objects can be freely placed in a simulated environment, eliminating the need to program and place NFC tags on physical objects. Also, arm and hand tracking is offered by the headset. This alternative setup helps to replicate experiments more effectively and follows in the current trends in using VR to ease the training for EMG prosthetics.

## Architecture

Leveraging LibEMG, we reuse the paradigm of a hardware-agnostic _OnlineDataHandler_ reading in live data from a UDP _streamer_.

The main process is responsible for:

- Loading a model
- Reading in the data from _OnlineDataHandler_
- Outputting gesture intentions, TBD: define if they are sent to ROS directly, TCP, UDP, etc
- If runtime labels are generated, use them to fine-tune (retrain from scratch?) on-the-fly the model

After an initial screen guided training step (or not?),

## Ideas

- execute a python script via ssh upon scanning an NFC tag with cellphone
- create object classes (shapes?) which are associated with metadata such as grip types
- program the NFC tags with the classes and stick them on objects
- when an NFC tag is scanned, label the next X s of data
- even better: label until significant IMU data is detected, meaning the arm is
- for each labelled "object type", use the IMU data to guess the correct label: eg a cup from the top is a "hand open", from the side "chuck grip (pinch)" (handle) or "power grip"

## Resources

- [LibEMG doc](https://libemg.github.io/libemg/#)
- [LibEMG gesture library](https://github.com/libemg/LibEMGGestures)
