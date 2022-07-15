from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

import numpy as np


class Command(IntEnum):
    LEFT = 0
    NO_DIRECTION = 1
    RIGHT = 2

    def as_action(self) -> np.ndarray:
        action_dict: Dict[Command, List[float]] = {
            Command.LEFT: [-1, 0.5, 0],
            Command.NO_DIRECTION: [0, 0.5, 0],
            Command.RIGHT: [1, 0.5, 0],
        }
        return np.array(action_dict[self])

@dataclass
class EvaluatedCommand:
    state: List[np.ndarray]
    command: Command
    next_state: List[np.ndarray]
    reward: float