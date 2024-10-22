# Self-supervised EMG gesture recognition

This project investigates the implementation of a self-supervised EMG gesture recognition system with the help of contextual information in the scope of interacting with day-to-day objects.

## Set up

Clone this repository, then:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e ./
```

## Usage

Only the `GAME` step requires the VR game. Normally, you'll want to execute them one at a time sequentially. It's tedious.

For VR experiments, first set your desired configurations in Config's `init` (`experiment/config.py`) and then set the _game stage_ `experiment/main.py` to the desired one. Launch the stage with:

`python3 experiment/main.py`

## VR Project

The VR project (Unity game) is only needed for the _Game_ stage. The Python server is responsible for generating live EMG gesture predictions which are retrieved by Unity.

1. Launch the Python experiment with `ExperimentStage.GAME` stage. It'll wait for Unity to start up
2. Launch the Unity game. See the [VR project](https://github.com/gabrielpgagne/VrGameRFID/tree/gab) repo's **gab** branch
3. In the game, launch the VR experiment by pressing the `O` key (start object grabbing task) and the `Space` key (start the timer and save scene metadata to disk).
4. When you're done, close the Unity Game. Python should automatically exit too.

All experiment logs are stored somewhere under `data/`. Unity logs are under `data/unity/` with the experiment start timestamp as filename. Python logs are stored in `data/<subject_id>/<sensor>/`.

See the VR project's readme for more details about the Unity side.

## Developing

This project also provides a somewhat generic library, located in `nfc_emg/`. Most things are self-explanatory or documented with docstrings.

## Resources

- [LibEMG doc](https://libemg.github.io/libemg/#)
- [LibEMG gesture library](https://github.com/libemg/LibEMGGestures)
