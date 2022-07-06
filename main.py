from enum import IntEnum
from typing import Dict, List

from gym.envs.box2d import CarRacing
import torch
from torch import nn
import torch.nn.functional as fn
import torch.optim as optim
import numpy as np

def print_scores(scores: torch.tensor):
    all_commands: List[Command] = [command for command in Command]
    for i in range(len(all_commands)):
        print(f"{str(all_commands[i])}: {round(float(scores[i]), 2)}")

class Command(IntEnum):
    LEFT = 0
    LEFT_MEDIUM = 1
    LEFT_BARELY = 2
    STRAIGHT = 3
    RIGHT_BARELY = 4
    RIGHT_MEDIUM = 5
    RIGHT = 6
    BRAKE = 7
    NO_GAS = 8
    MEDIUM_GAS = 9

def get_action(command: Command) -> np.ndarray:
    action_dict: Dict[Command, List[float]] = {
        Command.LEFT : [-1, 1, 0],
        Command.LEFT_MEDIUM : [0.6, 1, 0],
        Command.LEFT_BARELY : [0.3, 1, 0],
        Command.STRAIGHT : [0, 1, 0],
        Command.RIGHT_BARELY : [0.3, 1, 0],
        Command.RIGHT_MEDIUM : [0.6, 1, 0],
        Command.RIGHT : [1, 1, 0],
        Command.BRAKE : [0, 0, 1],
        Command.NO_GAS : [0, 0, 0],
        Command.MEDIUM_GAS : [0, 0.5, 0]
    }
    return np.array(action_dict[command])

class QNetwork(nn.Module):
    def __init__(self):
        super(QNetwork, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, (5,5))
        self.conv2 = nn.Conv2d(6, 16, (5,5))
        self.fc1 = nn.Linear(16 * 21 * 21, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        pool: nn.MaxPool2d = nn.MaxPool2d(2, 2)
        x = pool(fn.relu(self.conv1(x)))
        x = pool(fn.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = fn.relu(self.fc1(x))
        x = fn.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def train_model(self, states: List[np.ndarray], rewards: List[float], commands: List[Command]):
        # Update Model
        in_greychannel: List[np.ndarray] = [np.average(state, axis=2).astype(np.float32) for state in states]
        as_tensors: List[torch.tensor] = [torch.tensor(correct_shape) for correct_shape in in_greychannel]
        correct_shape_for_model: List[torch.tensor] = [tensor[None, ...] for tensor in as_tensors]

        # Create Target Result
        target_tensors: List[torch.tensor] = []
        for state, reward, command in zip(states, rewards, commands):
            target_tensor: torch.tensor = torch.FloatTensor([0.3 for i in range(10)])
            target_tensor[int(command)] = 0
            if reward > 6:
                target_tensor[int(command)] = 1
            target_tensors.append(target_tensor)

        # Train Model
        for i in range(30):
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adam(q_learner.parameters(), lr=0.01)
            scores: torch.tensor = self(torch.stack(correct_shape_for_model))
            loss = criterion(scores, torch.stack(target_tensors))
            loss.backward()
            optimizer.step()

            # Print Information
            print("Model Update:")
            print(f"Loss: {loss}")
            print("")

    def predict(self, state: np.ndarray) -> Command:
        correct_shape: np.ndarray = np.average(state, axis=2).astype(np.float32)
        as_tensor: torch.tensor = torch.tensor(correct_shape)
        as_tensor = as_tensor[None, None, ...]
        scores: torch.tensor = self(as_tensor)
        as_command: Command = Command(int(torch.argmax(scores)))
        return as_command


# Init Car Racing
car_racing: CarRacing = CarRacing()
car_racing.reset()
all_commands: List[Command] = [command for command in Command]

# Init Network
q_learner: QNetwork = QNetwork()
states: List[torch.tensor] = []
rewards: List[float] = []
commands: List[Command] = []

while True:
    car_racing.reset(seed=0)

    # Drive until Leaving Track
    current_reward: int = 0
    predicted_command: Command = Command.STRAIGHT
    i: int = 0
    while current_reward >= 0 or i % 10 != 0:
        if i % 10 == 0:
            predicted_command = q_learner.predict(car_racing.state)
            states.append(car_racing.state)
            rewards.append(current_reward)
            commands.append(predicted_command)
            current_reward = 0
        as_action: np.ndarray = get_action(predicted_command)
        current_state, reward, done, info = car_racing.step(as_action)
        current_reward += reward
        i += 1
        car_racing.render(mode="human")
    rewards.append(current_reward)
    rewards = rewards[1:]

    # Train Model
    q_learner.train_model(states, rewards, commands)






