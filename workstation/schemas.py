from enum import IntEnum


class ArmControl(IntEnum):
    NEUTRAL = 0
    LEFT = 1  # horizontal (x)
    RIGHT = 2  # horizontal (-x)
    UP = 3  # horizontal (y)
    DOWN = 4  # horizontal (-y)
    SUPINATION = 5  # vertical (z)
    PRONATION = 6  # vertical (-z)


class WristControl(IntEnum):
    NEUTRAL = 0
    FLEXION = 1  # towards body, clockwise rotation
    EXTENSION = 2  # away from body, counter-clockwise rotation
    ABDUCTION = 3  # down, end-effector towards ground
    ADDUCTION = 4  # up, end-effector towards sky


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
