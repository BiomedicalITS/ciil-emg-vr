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
    """Convert a dictionary to an enum object from this module

    Args:
        d (dict): Must have key "arm", "wrist" or "gripper" with a value corresponding to the enum

    Returns:
        IntEnum: The enum object
    """
    if "arm" in d.keys():
        return ArmControl(d["arm"])
    elif "wrist" in d.keys():
        return WristControl(d["wrist"])
    elif "gripper" in d.keys():
        return GripperControl(d["gripper"])


def to_dict(e: IntEnum):
    """Convert an enum object to a dictionary

    Args:
        e (IntEnum): An object from this module

    Returns:
        dict: The dictionary
    """
    if e in ArmControl:
        return {"arm": int(e)}
    elif e in WristControl:
        return {"wrist": int(e)}
    elif e in GripperControl:
        return {"gripper": int(e)}


if __name__ == "__main__":
    o = WristControl(1)
    print(o.name)
    print(o in WristControl)
    print(o in GripperControl)
    dico = to_dict(o)
    obj = from_dict(dico)
    print(dico, obj)
