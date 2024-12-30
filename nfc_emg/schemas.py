from enum import IntEnum


class GenericControl(IntEnum):
    def __eq__(self, other):
        return self is other

    def __ne__(self, value: object) -> bool:
        return not self == value


class ArmControl(GenericControl):
    NEUTRAL = 0
    LEFT = 1
    """ horizontal (x)"""
    RIGHT = 2
    """horizontal (-x)"""
    UP = 3
    """horizontal (y)"""
    DOWN = 4
    """horizontal (-y)"""
    SUPINATION = 5
    """vertical (z)"""
    PRONATION = 6
    """vertical (-z)"""


class WristControl(GenericControl):
    NEUTRAL = 0
    FLEXION = 1
    """towards body, clockwise rotation"""
    EXTENSION = 2
    """away from body, counter-clockwise rotation"""
    ABDUCTION = 3
    """down, end-effector towards ground"""
    ADDUCTION = 4
    """up, end-effector towards sky"""


class GripperControl(GenericControl):
    NEUTRAL = 0
    CLOSE = 1
    OPEN = 2


class ObjectShape(IntEnum):
    BUTTON = 0  # most likely index extension
    CYLINDRICAL_THIN = 1  # most likely pinch or can be grip
    CYLINDRICAL_THICK = 2  # most likely grip
    FLAT_THIN = 3  # most likely chuck grip or hand open

    @staticmethod
    def get_possible_gestures(name_to_cid: dict, val: int):
        if val == ObjectShape.BUTTON:
            return (name_to_cid["Index_Extension"],)
        elif val == ObjectShape.CYLINDRICAL_THIN:
            return (name_to_cid["Index_Pinch"], name_to_cid["Hand_Close"])
        elif val == ObjectShape.CYLINDRICAL_THICK:
            return (name_to_cid["Hand_Close"], name_to_cid["Index_Pinch"])
        elif val == ObjectShape.FLAT_THIN:
            return (name_to_cid["Chuck_Grip"], name_to_cid["Hand_Open"])
        else:
            return tuple()


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
    else:
        raise KeyError("No valid key in dictionary")


def to_dict(e: GenericControl):
    """Convert an enum object to a dictionary

    Args:
        e (IntEnum): An object from this module

    Returns:
        dict: The dictionary
    """
    if isinstance(e, ArmControl):
        return {"arm": int(e)}
    elif isinstance(e, WristControl):
        return {"wrist": int(e)}
    elif isinstance(e, GripperControl):
        return {"gripper": int(e)}
    else:
        raise ValueError("Invalid enum type")


POSE_TO_NAME = {
    "Ne": "No_Motion",
    "H1": "Index_Pinch",
    "H2": "Hand_Close",
    "H3": "Chuck_Grip",
    "T1": "Index_Extension",
    "T2": "Wrist_Flexion",
    "T4": "Hand_Open",
}

NAME_TO_SHORT = {
    "Chuck_Grip": "CG",
    "Hand_Close": "HC",
    "Hand_Open": "HO",
    "Index_Extension": "IE",
    "Index_Pinch": "IP",
    "No_Motion": "HR",
    "Wrist_Extension": "WE",
    "Wrist_Flexion": "WF",
}

if __name__ == "__main__":
    # Test todict and fromdict for every class
    o = ArmControl.NEUTRAL
    d = to_dict(o)
    assert d == {"arm": 0}
    assert from_dict(d) == o

    o = WristControl.ABDUCTION
    d = to_dict(o)
    assert d == {"wrist": 3}
    assert from_dict(d) == o

    o = GripperControl.OPEN
    d = to_dict(o)
    assert d == {"gripper": 2}
    assert from_dict(d) == o

    # Test equality
    o = ArmControl.NEUTRAL
    o2 = ArmControl(0)
    assert o == o2

    i = 0
    i2 = 1
    assert ArmControl(i) != ArmControl(i2)

    o = WristControl.ABDUCTION
    o2 = from_dict({"wrist": 3})
    assert o == o2

    o = GripperControl.NEUTRAL
    i = 0
    assert int(o) == i

    o = GripperControl.NEUTRAL
    o2 = ArmControl.NEUTRAL
    assert o != o2

    o = GripperControl.NEUTRAL
    o2 = ArmControl.NEUTRAL
    assert int(o) == int(o2)

    print("All test cases passed!")
