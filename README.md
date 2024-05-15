# Self-supervised EMG gesture recognition

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
