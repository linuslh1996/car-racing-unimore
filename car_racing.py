from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

import cv2
import numpy as np
import torch


class Command(IntEnum):
    LEFT = 0
    NO_DIRECTION = 1
    RIGHT = 2
    GAS = 3
    BRAKE = 4

    def as_action(self) -> np.ndarray:
        action_dict: Dict[Command, List[float]] = {
            Command.LEFT: [-1, 0.5, 0],
            Command.NO_DIRECTION: [0, 0.5, 0],
            Command.RIGHT: [1, 0.5, 0],
            Command.GAS: [0, 1, 0],
            Command.BRAKE: [0, 0, 0.8]
        }
        return np.array(action_dict[self])

    @staticmethod
    def all_commands() -> list:
        return [command for command in Command]


@dataclass
class EvaluatedCommand:
    state: List[np.ndarray]
    command: Command
    next_state: List[np.ndarray]
    rewards: List[float] # The next 5 Rewards after the action
    log_probability: float

def state_to_tensor(input_state: List[np.ndarray]) -> torch.Tensor:
    greyscale_images: List[np.ndarray] = [cv2.cvtColor(state, cv2.COLOR_BGR2GRAY) for state in input_state]
    normalized: List[np.ndarray] = [greyscale_image / 255.0 for greyscale_image in greyscale_images]
    as_tensors: List[torch.Tensor] = [torch.Tensor(normal.astype(np.float32)) for normal in normalized]
    as_tensor: torch.Tensor = torch.stack(as_tensors)
    return as_tensor