from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import List, Tuple

import cv2
import torch
from torch import nn
import torch.nn.functional as fn
import torch.optim as optim
import numpy as np
from typing import Dict

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

def custom_loss_function(scores: torch.Tensor, target: torch.Tensor, performed_commands: List[EvaluatedCommand]) -> torch.Tensor:
    weights = torch.zeros(target.shape)
    for i in range(len(scores)):
        weights[i][int(performed_commands[i].command)] = 1
    return (weights * (scores - target) **2).sum() / len(performed_commands)


class QNetwork(nn.Module):
    def __init__(self, weights_path: Path):
        # Build Network
        super(QNetwork, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, (7,7), stride=(3,3))
        self.conv2 = nn.Conv2d(6, 12, (4,4))
        self.fc1 = nn.Linear(12 * 6 * 6, 216)
        self.fc2 = nn.Linear(216, len([command for command in Command]))

        # Init Cache for Printing
        self.weights_path: Path = weights_path
        self.current_score: torch.Tensor = torch.zeros(0)
        self.current_evaluated_commands: List[EvaluatedCommand] = []
        self.current_loss: torch.Tensor = torch.Tensor(0)

    def forward(self, x):
        pool: nn.MaxPool2d = nn.MaxPool2d(2, 2)
        x = pool(fn.relu(self.conv1(x)))
        x = pool(fn.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = fn.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def create_target_scores(self, evaluated_commands: List[EvaluatedCommand], scores: torch.Tensor, gamma: float, target_network) -> List[torch.Tensor]:
        target_tensors: List[torch.Tensor] = []
        for i, evaluated_command in enumerate(evaluated_commands):
            target_tensor: torch.Tensor = torch.clone(scores[i])
            _, best_score_next_state, _ = target_network.predict(evaluated_command.next_state)
            target_tensor[int(evaluated_command.command)] = evaluated_command.reward + gamma * best_score_next_state
            target_tensors.append(target_tensor)
        return target_tensors

    def state_to_tensor(self, input_state: List[np.ndarray]) -> torch.Tensor:
        greyscale_images: List[np.ndarray] = [cv2.cvtColor(state, cv2.COLOR_BGR2GRAY) for state in input_state]
        normalized: List[np.ndarray] = [greyscale_image / 255.0 for greyscale_image in greyscale_images]
        as_tensors: List[torch.Tensor] = [torch.Tensor(normal.astype(np.float32)) for normal in normalized]
        as_tensor: torch.Tensor = torch.stack(as_tensors)
        return as_tensor

    def train_model(self, evaluated_commands: List[EvaluatedCommand], learning_rate: float, gamma: float, target_network):
        # Update Model
        input_states: List[torch.Tensor] = [self.state_to_tensor(command.state) for command in evaluated_commands]

        # Train Model
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        for i in range(1):
            # Calculate Loss & Update Weights
            scores: torch.Tensor = self(torch.stack(input_states))
            target_scores: List[torch.Tensor] = self.create_target_scores(evaluated_commands, scores, gamma, target_network)
            criterion = nn.MSELoss()
            loss = criterion(scores, torch.stack(target_scores))
            loss.backward()
            optimizer.step()

            # Update Cache
            self.current_evaluated_commands = evaluated_commands
            self.current_loss = loss
            self.current_score = scores

    def predict(self, state: List[np.ndarray]) -> Tuple[Command, float, torch.Tensor]:
        as_tensor: torch.Tensor = self.state_to_tensor(state)
        as_tensor = as_tensor[None, ...]
        scores: torch.Tensor = self(as_tensor)
        as_command: Command = Command(int(torch.argmax(scores)))
        score: float = float(torch.max(scores))
        return as_command, score, scores

    def set_weights(self, other_network):
        self.load_state_dict(other_network.state_dict())

    def save_model(self, episode: int):
        torch.save(self.state_dict(), self.weights_path / f"weights_{episode}.pt")

    def load_model(self, episode: int):
        self.load_state_dict(torch.load(self.weights_path / f"weights_{episode}.pt"))

    def print_scores(self, scores: torch.Tensor, evaluated_commands: List[EvaluatedCommand], loss: torch.Tensor, gamma: float, target_network):
        all_commands: List[Command] = [command for command in Command]
        for j in range(scores.shape[0]):
            _, best_score_next_state, _ = target_network.predict(evaluated_commands[j].next_state)
            print(f"{str(evaluated_commands[j].command).split('.')[1]} performed: {round(evaluated_commands[j].reward + gamma * best_score_next_state, 2)}        ", end=" ")
            for i in range(len(all_commands)):
                print(f"{str(all_commands[i]).split('.')[1]}: {round(float(scores[j][i]), 2)}", end=" ")
            print("")
        print("")
        print(f"Loss: {round(float(loss), 2)}")
