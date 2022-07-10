import random
from enum import IntEnum
from typing import Dict, List, Tuple

from gym.envs.box2d import CarRacing
import torch
from torch import nn
import torch.nn.functional as fn
import torch.optim as optim
import numpy as np
import cv2

class Command(IntEnum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2

def print_scores(scores: torch.tensor, rewards: List[float], commands: List[Command]):
    all_commands: List[Command] = [command for command in Command]
    for j in range(scores.shape[0]):
        print(f"Reward: {round(rewards[j], 2)}, {str(commands[j])}")
        for i in range(len(all_commands)):
            print(f"{str(all_commands[i]).split('.')[1]}: {round(float(scores[j][i]), 2)}", end=" ")
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

    def create_target_scores(self, scores: torch.tensor, states: List[np.ndarray], rewards: List[float], commands: List[Command]) -> List[torch.tensor]:
        # Create Target Result
        target_tensors: List[torch.tensor] = []
        for i, (state, reward, command) in enumerate(zip(states, rewards, commands)):
            target_tensor: torch.tensor = torch.clone(scores[i])
            target_tensor[int(command)] = reward
            target_tensors.append(target_tensor)
        return target_tensors

    def train_model(self, states: List[np.ndarray], rewards: List[float], commands: List[Command]):
        # Update Model
        correct_shape: List[np.ndarray] = [np.moveaxis(state, -1, 0).astype(np.float32) for state in states]
        as_tensors: List[torch.tensor] = [torch.tensor(correct_shape) for correct_shape in correct_shape]

        # Train Model
        loss = ""
        scores = ""
        optimizer = optim.Adam(q_learner.parameters(), lr=0.0001)
        for i in range(1):
            scores = self(torch.stack(as_tensors))
            target_scores: List[torch.tensor] = self.create_target_scores(scores, states, rewards, commands)
            criterion = nn.MSELoss()
            loss = criterion(scores, torch.stack(target_scores))
            loss.backward()
            optimizer.step()

        # Print Information
        print("Model Update:")
        print_scores(scores, rewards, commands)
        print(f"Loss: {loss}")

    def predict(self, state: np.ndarray) -> Tuple[Command, float, torch.tensor]:
        correct_shape: np.ndarray = np.moveaxis(state, -1, 0).astype(np.float32)
        as_tensor: torch.tensor = torch.tensor(correct_shape)
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
states: List[torch.tensor] = []
rewards: List[float] = []
commands: List[Command] = []
epsilon: float = 1.0

while True:
    # Start Track Again
    car_racing.reset(seed=0)

    # Drive on Track until leaving Track
    current_reward: int = 0
    while current_reward >= 0:
        # Make Decision Where to Drive
        current_reward = 0
        command, score, scores = q_learner.predict(car_racing.state)
        predicted_command: Command = command
        if random.random() < epsilon:
            predicted_command = random.choice(all_commands)
        as_action: np.ndarray = get_action(predicted_command)
        for i in range(20):
            current_state, reward, done, info = car_racing.step(as_action)
            current_reward += reward
            #car_racing.render(mode="human")
        states.append(car_racing.state)
        rewards.append(current_reward)
        commands.append(predicted_command)

        # Train Model
        if len(states) > BATCH_SIZE:
            as_zip = list(zip(states, rewards, commands))
            sampled = random.sample(list(as_zip), BATCH_SIZE)
            states_sampled, rewards_sampled, commands_sampled = zip(*sampled)
            q_learner.train_model(states_sampled, rewards_sampled, commands_sampled)
            epsilon *= 0.9999
            print(f"Epsilon: {epsilon}")
            print("")






