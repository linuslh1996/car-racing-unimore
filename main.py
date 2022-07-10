import random
from dataclasses import dataclass
from enum import IntEnum
from typing import Dict, List, Tuple

from gym.envs.box2d import CarRacing
import torch
from torch import nn
import torch.nn.functional as fn
import torch.optim as optim
import numpy as np

class Command(IntEnum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2

@dataclass
class EvaluatedCommand:
    state: np.ndarray
    command: Command
    next_state: np.ndarray
    reward: float

def print_scores(scores: torch.tensor, evaluated_commands: List[EvaluatedCommand]):
    all_commands: List[Command] = [command for command in Command]
    for j in range(scores.shape[0]):
        print(f"{str(evaluated_commands[j].command)} performed: {round(evaluated_commands[j].reward, 2)}")
        for i in range(len(all_commands)):
            print(f"{str(all_commands[i]).split('.')[1]}: {round(float(scores[j][i]), 2)}", end=" ")
        print("")
        print("")
    print("")

def get_action(command: Command) -> np.ndarray:
    action_dict: Dict[Command, List[float]] = {
        Command.LEFT : [-1, 0.5, 0],
        Command.STRAIGHT : [0, 0.5, 0],
        Command.RIGHT : [1, 0.5, 0]
    }
    return np.array(action_dict[command])

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, (7,7), stride=3)
        self.conv2 = nn.Conv2d(6, 12, (4,4))
        self.fc1 = nn.Linear(12 * 6 * 6, 216)
        self.fc2 = nn.Linear(216, 3)

    def forward(self, x):
        pool: nn.MaxPool2d = nn.MaxPool2d(2, 2)
        x = pool(fn.relu(self.conv1(x)))
        x = pool(fn.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = fn.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def create_target_scores(self, evaluated_commands: List[EvaluatedCommand], scores: torch.tensor) -> List[torch.tensor]:
        target_tensors: List[torch.tensor] = []
        for i, evaluated_command in enumerate(evaluated_commands):
            target_tensor: torch.tensor = torch.clone(scores[i])
            _, best_score_next_state, _ = self.predict(evaluated_command.next_state)
            target_tensor[int(command)] = evaluated_command.reward
            target_tensors.append(target_tensor)
        return target_tensors

    def state_to_tensor(self, input_state: np.ndarray) -> torch.tensor:
        with_color_channel_in_correct_axis: np.ndarray = np.moveaxis(input_state, -1, 0).astype(np.float32)
        as_tensor: torch.tensor = torch.tensor(with_color_channel_in_correct_axis)
        return as_tensor

    def train_model(self, evaluated_commands: List[EvaluatedCommand]):
        # Update Model
        input_states: List[torch.tensor] = [self.state_to_tensor(command.state) for command in evaluated_commands]

        # Train Model
        loss = ""
        scores = ""
        optimizer = optim.Adam(q_learner.parameters(), lr=0.0001)
        for i in range(1):
            scores = self(torch.stack(input_states))
            target_scores: List[torch.tensor] = self.create_target_scores(evaluated_commands, scores)
            criterion = nn.MSELoss()
            loss = criterion(scores, torch.stack(target_scores))
            loss.backward()
            optimizer.step()

        # Print Information
        print("Model Update:")
        print_scores(scores, evaluated_commands)
        print(f"Loss: {loss}")

    def predict(self, state: np.ndarray) -> Tuple[Command, float, torch.tensor]:
        as_tensor: torch.tensor = self.state_to_tensor(state)
        as_tensor = as_tensor[None, ...]
        scores: torch.tensor = self(as_tensor)
        as_command: Command = Command(int(torch.argmax(scores)))
        score: float = float(torch.max(scores))
        return as_command, score, scores

# Init Car Racing
car_racing: CarRacing = CarRacing()
car_racing.reset()
all_commands: List[Command] = [command for command in Command]

# Init Network
q_learner: QNetwork = QNetwork()
BATCH_SIZE: int = 64
evaluated_commands: List[EvaluatedCommand] = []
epsilon: float = 1.0

while True:
    # Start Track Again
    car_racing.reset(seed=0)

    # Drive on Track until leaving Track
    current_reward: int = 0
    while current_reward >= 0:
        # Make Decision Where to Drive
        current_reward = 0
        beginning_state: np.ndarray = car_racing.state
        command, score, scores = q_learner.predict(beginning_state)
        performed_command: Command = command
        if random.random() < epsilon:
            performed_command = random.choice(all_commands)
        as_action: np.ndarray = get_action(performed_command)
        end_state: np.ndarray = car_racing.state
        for i in range(10):
            end_state, reward, done, info = car_racing.step(as_action)
            current_reward += reward
            #car_racing.render(mode="human")
        evaluated_commands.append(EvaluatedCommand(beginning_state, performed_command, end_state, current_reward))

        # Train Model
        if len(evaluated_commands) > BATCH_SIZE:
            sampled = random.sample(evaluated_commands, BATCH_SIZE)
            q_learner.train_model(sampled)
            epsilon *= 0.9995
            print(f"Epsilon: {epsilon}")
            print("")






