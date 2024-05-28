import numpy as np
from serial.tools import list_ports


vals = np.array(
    [
        [0, 1],
        [0, 1],
        [1, 0],
        [1, 1],
        [2, 2],
    ]
)
test = np.array([0, 1])

print(vals == test)
pos = np.argwhere((vals == test).all(axis=1))
print(pos)

ps = list_ports.comports()
print([(p.device, p.description) for p in ps])
