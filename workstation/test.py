import serial
from serial.tools import list_ports

ps = list_ports.comports()
print([(p.device, p.description) for p in ps])
