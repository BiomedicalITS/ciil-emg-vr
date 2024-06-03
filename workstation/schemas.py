from enum import IntEnum


class ArmControl(IntEnum):
    NEUTRAL = 0
    LEFT = 1
    RIGHT = 2
    UP = 3
    DOWN = 4
    SUPINATION = 5
    PRONATION = 6


class WristControl(IntEnum):
    NEUTRAL = 0
    FLEXION = 1  # towards body
    EXTENSION = 2  # away from body
    ABDUCTION = 3  # down
    ADDUCTION = 4  # up


class GripperControl(IntEnum):
    NEUTRAL = 0
    CLOSE = 1
    OPEN = 2


def from_dict(d: dict):
    if "arm" in d.keys():
        return ArmControl(d["arm"])
    elif "wrist" in d.keys():
        return WristControl(d["wrist"])
    elif "gripper" in d.keys():
        return GripperControl(d["gripper"])


def to_dict(e: IntEnum):
    if e in ArmControl:
        return {"arm": e}
    elif e in WristControl:
        return {"wrist": e}
    elif e in GripperControl:
        return {"gripper": e}


if __name__ == "__main__":
    o = WristControl(1)
    print(o.name)
    print(o in WristControl)
    print(o in GripperControl)
    print(to_dict(o))
