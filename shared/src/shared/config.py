from typing import Final
from dataclasses import dataclass

# Motor Settings
MAX_SPEED: Final = 0.3
STOP_SPEED: Final = -1.0
STEERING_CENTER: Final = 80
MAX_LEFT: Final = 50
MAX_RIGHT: Final = 50

# Joystick Settings
A: Final = 0
B: Final = 1
X: Final = 2
Y: Final = 3
RT: Final = 5
LEFTSTICK: Final = 0

# Camera Settings
ROTATE: Final = 2 # 90度を何回ローテーションするか、２の場合１８０度
