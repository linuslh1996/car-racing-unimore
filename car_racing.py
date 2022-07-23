from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List

import cv2
import numpy as np
import torch
from gym.envs.box2d import CarRacing

STEP_SIZE: int = 10


class Command(IntEnum):
    LEFT = 0
    NO_DIRECTION = 1
    RIGHT = 2
    GAS = 3
    BRAKE = 4

    def as_action(self) -> np.ndarray:
        action_dict: Dict[Command, List[float]] = {
            Command.LEFT: [-1, 0, 0],
            Command.NO_DIRECTION: [0, 0, 0],
            Command.RIGHT: [1, 0, 0],
            Command.GAS: [0, 1, 0],
            Command.BRAKE: [0, 1, 0.5]
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

class CustomRacing:

    def __init__(self, current_episode: int):
        self._done = False
        self._accumulated_reward = 0
        self._car_racing: CarRacing = CarRacing()
        self._car_racing.reset()
        self._negative_rewards_in_a_row = 0
        self._current_time = 0
        self._current_state = self._car_racing.state
        self._current_episode = current_episode

    def reset(self, seed: int=None):
        self._car_racing.reset(seed=seed)
        self._done = False
        self._accumulated_reward = 0
        if self._current_time > 0:
            self._current_episode += 1
        self._current_time = 0

    def perform_step(self, command: Command, render: bool=False) -> float:
        step_reward: float = 0
        for i in range(STEP_SIZE):
            end_state, reward, finished_track, info = self._car_racing.step(command.as_action())
            step_reward += reward
            self._current_time += 1
            if render:
                self._car_racing.render(mode="human")
        self._negative_rewards_in_a_row = self._negative_rewards_in_a_row + 1 if step_reward < 0 else 0
        self._accumulated_reward += step_reward
        self._current_state = self._car_racing.state
        return step_reward

    def done(self) -> bool:
        return self._accumulated_reward > 500

    def current_state(self) -> np.ndarray:
        return self._current_state

    def current_episode(self) -> int:
        return self._current_episode

    def out_of_track(self):
        return self._negative_rewards_in_a_row >= 30 / STEP_SIZE and self._current_time > 50