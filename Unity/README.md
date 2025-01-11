# EMaGerVRStartup

## Setup

First, see [nfc-emg](https://github.com/BiomedicalITS/nfc-emg) project.

```bash
git clone https://github.com/ThomasLabbe01/EMaGerVRStartup.git
cd VrGameRFID
git switch gab
```

### VR headset

This has only been tested with Quest 2 and 3. Enable Quest Link, which allows you to stream PC VR games to the headset. Then, when you launch the Unity game, it should automatically launch it in the Quest.

### Setup Unity project

- Open the project in Unity Editor. If prompted, install **Unity 2022.3.4 or Unity 2022.3.5**.
- In _Project_, click on _Scenes_. Drag-n-drop SGTGab and DemoGab into the _Hierarchy_. Delete the default scene.
- In _Hierarchy_, Set the scene `DemoGab` as active (right click -> Set Active Scene) and `SGTGab` as inactive (right click -> Unload). SGT is done purely in Python.
- For development, in both scenes, click and WebGL Rig -> check the box in the top of the right pane. For VR, uncheck it.
  - WebGL Rig allows to interact with the game with keyboard and mouse. Upon launching the game, press **F12** to enable KB&M control. Then, you can move with arrows and look around with the mouse.

## Running

### Python Server

Run NfcEmg's `experiment/main.py`. If it is in `GAME` mode, it will wait for Unity to send a "READY" message on **12350/udp**.

### Run the VR game

**NOTE** This step is skipped if you set _DemoGab_ as the active scene.

Upon launching the game (Play button in Unity Editor), you should be in front of a configuration panel. From here, you can do an SGT (`LCtrl + Enter`, then `LShift + Enter`) or skip straight into the actual game (`Right Shift + Enter`).

#### VR SGT

- Press the _Play_ button right over the preview window
- Focus into the game by clicking in the preview window
- Press `LCtrl + Enter` to quickly set up the SGT.
- Choose the hand gestures in `Assets/resources`.
- Select the data save location in `EMaGerVRStartup/Data`.
- Press `LShift + Enter` to start the SGT on the right.

- Wait for the classifier to output data in the Python console.
- Press `Right Shift + Enter` to switch to the correct scene.

#### Grabbing tasks

- `Enter` toggles the on/off state of the table and the objects.
- Instructions in text will appear with the options:
  - F for freedom mode, where ghost grips appear when near objects
  - O for testing mode
  - V for _visual_ testing mode
  - ...
- `Esc` will exit whatever mode you're in and go back to the "waiting" mode with the instructions


## About

### Research question

EMG gesture recognition systems suffer from confounding factors such as electrode shift and changes in muscle activity patterns over time. To address this, we aim to recalibrate the model in real-time with the help of contextual information used to generate live training examples.

### Methods

1. Install the BioArmband on the forearm and wait 5 minutes
2. Launch Python server
3. Put on the VR headset
4. Launch the VR game and make sure the hand tracking is working properly
5. Do an initial Screen Guided Training of 5 repetitions of 3s
6. a (optional) let the user get acquainted with the scene in Freedom mode (F)
6. b  Enter the game in Visual (?) testing mode and do every item prompt
7. Do an exit SGT of 2 repetitions of 3s

All in all, the experiment should take ~15 minutes per subject.

The initial SGT-only model will be validated and  tested on 2 repetition (repetitions 4 and 5, respectively). This will give an initial performance benchmark for every subject.

Finally, both the SGT-only model and self-supervised model will be evaluated on the 2 post-test repetitions.

For both testing cases, typical machine learning metrics will be used to assess the performance.

## Quick Description

EMaGerVRStartup is an innovative VR project that integrates EMG (Electromyography) signal processing to create an immersive experience for users. This project includes:

1. **Python Server**: See [NfcEmg](https://github.com/BiomedicalITS/nfc-emg) project
2. **EMGRawReader**: The EMGRawReader GameObject (`Assets/Scripts/TCPManagers/EMGRawReader.cs`) receives live predictions from the Python server on port **12347/udp**
3. **Screen Guided Training in VR**: A VR-based guided training session helps users learn and perform specific gestures accurately.
4. **Hand Gestures Props**: Props simulate a prosthesis, allowing users to practice and perfect hand gestures.
5. **WebRL Rig**: A 3D capsule that can substitute for a person, enabling the game to be played (with limitations) without a VR headset
6. **Quick Game Setup Canvas**: The canvas includes a table and an apple. The apple can be grabbed and is influenced by gravity, providing a realistic interaction experience.
7. **Hand Poses via Color Code**: Color-coded hand poses enable players to grab objects by performing the correct gestures.

## Notes

### Python-Unity relationship

The game requires the Python server to be already running and waiting for a TCP connection. The Python server is the central repository for documentation and the "brains" of the experiment. The VR game serves as the experimentation context. The game **will** launch without the Python server, but you'll miss out on the live classification, SGT data recording, etc.

### Disabling live classification

In Unity's Hierarchy -> DemoGab -> (left click) test -> (Inspector) EMG \[ ].

This is useful when you want to manually set the active gesture with the keypad numbers 1-9. See [Gestures](#gestures) for what each color means.

### Objects

T3 was removed from the allowed gestures since it does not resemble any hand gesture (NFC VR project). If you change the list of items to use, make sure to include them in the `test` object's `ObjectsToSave` variable under _Hierarchy_.

A good short balanced selection of objects is, which you can hardcode in `Assets/Scripts/General/TestVsFreedom.cs`'s `poses_to_do`:

1. Apple
2. Frying pan
3. Key
4. Chicken leg
5. Cherry
6. Smartphone

| Object      | Allowed gestures | Pose ID start |
| ----------- | ---------------- | ------------- |
| Apple       | H1,T2            | 0             |
| Bowl        | H2,T2,T3         | 2             |
| Xmas lolli  | H2,H3            | 5             |
| Chicken leg | H2,H3,T2         | 7             |
| Crown       | H1,H4,T3         | 10            |
| Frying pan  | H2,H3,T4         | 13            |
| Lolli       | H2,H4            | 16            |
| Pot         | H2,T3,T4         | 18            |
| Pear        | H1,T2,           | 21            |
| Donut       | H4,T1            | 23            |
| Key         | H2,H4,T1         | 25            |
| Fish        | H1,T2            | 28            |
| Skate       | H3,T4            | 30            |
| Wheel       | T1,T3,T4         | 32            |
| Hat         | H1,T3,T4         | 35            |
| Tape        | H4,T1            | 38            |
| Pin         | H4,T2            | 40            |
| Smartphone  | H3,T4            | 42            |
| Cherry      | H1,T1,T2         | 44            |

### Gestures

NfcEmg's `config.py` file defines some functional gesture sets. They were mostly matched with the VR game's supported _poses_. The VR project supports 9 gestures:

| Gesture name | Color  | ID  | Looks like      | Mapped with Python? |
| ------------ | ------ | --- | --------------- | ------------------- |
| Neutral      | White  | 0   | No motion       | Yes                 |
| H1           | Yellow | 1   | Index pinch     | Yes                 |
| H2           | Orange | 2   | Power grip      | Yes                 |
| H3           | Red    | 3   | Chuck grip      | Yes                 |
| H4           | Pink   | 4   | Tri pinch       | No                  |
| T1           | Purple | 5   | Index extension | Yes                 |
| T2           | Blue   | 6   | Wrist flexion   | Yes                 |
| T3           | Cyan   | 7   | ???             | No                  |
| T4           | Green  | 8   | Hand open       | Yes                 |

H stands for "Hand" and T for "Tentacle". Tentacle is useful, for example, for amputees, which may not have the muscle memory to know "how" to do a hand gesture. Imagine if you were asked to flex an imaginary 6th finger on your left hand. With tentacles, you are asked to do an abstract gesture and it's up to you to interpret it.

The Python gestures were chosen to be similar to the VR game's poses. Below is the mapping, which is done dynamically in `EMGRawReader.cs`'s `ReceiveData` method. Make sure the VR SGT also selects these exact gestures in this order.

| LibEMG name     | Python ID | VR Name | VR ID | VR Color |
| --------------- | --------- | ------- | ----- | -------- |
| Chuck_Grip      | 0         | H3      | 3     | Red      |
| Hand_Close      | 1         | H2      | 2     | Orange   |
| Hand_Open       | 2         | T4      | 8     | Green    |
| Index_Extension | 3         | T1      | 5     | Purple   |
| Index_Pinch     | 4         | H1      | 1     | Yellow   |
| No_Motion       | 5         | Neutral | 0     | White    |
| Wrist_Extension | 6         | T3 *1   | 7     | Cyan     |
| Wrist_Flexion   | 7         | T2      | 6     | Blue     |
| "Rejected"      | -1        | Neutral | 0     | White    |

\*1 Mapped to No_Motion in VR code, which is ignored for sending context to the Python server. 0 is appended to the `prediction_map` in case Rejection is used, which returns -1 when a classification is rejected.

### Live context

Unity receives data from 2 UDP ports:

- **12347/udp** for OnlineEMGClassifier. **Unity acts as server**.
- **12350/udp** to read/write to NfcEmg's AdaptationManager, **Unity acting as client**.

In _Testing mode_ or _Visual Testing_, whenever the **right** hand is near an object, a _context_ packet is sent to the Python server via port `12350/udp`. Also, when the game exits, a `"Q"` is sent in the same port, which can be used to close the server application.

For every object, a set of grabbing _poses_ are defined (Hierarchy -> DemoGab -> test -> NewObjects v2 -> Any item -> Poses). From the Python's live classifier output and the mapping from Py to VR, Unity knows if we're currently _within context_ or _out-of-context_.

The context is formatted as `{P, N} X ... Xn`, where P means is _within context_ and N means it's _out-of-context_. X ... Xn are a space-separated list of possible gestures in the current context.

With this information, it's possible to train the model on its current prediction when _within context_. Out-of-context cases can be handled with more flexibility, which is out of scope for this project.
