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

OBJECT_TO_CONTEXT = {
    "Apple": [4, 7, -1],
    "FryingPan": [1, 0, 2],
    "Key": [1, 3, -1],
    "ChickenLeg": [1, 0, 7],
    "Cherry": [4, 3, 7],
    "Cheery": [4, 3, 7],
    "SmartPhone": [0, 2, -1],
}
