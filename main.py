import random
from dataclasses import dataclass
from enum import IntEnum
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import gym
from gym.envs.box2d import CarRacing
import torch
from torch import nn
import torch.nn.functional as fn
import torch.optim as optim
import numpy as np
import sys

GAMMA: float = 0.95
STEP_SIZE = 10
TARGET_NETWORK_UPDATE_FREQUENCY = 5 * STEP_SIZE
LEARNING_RATE = 0.001
BATCH_SIZE: int = 64
EPSILON_DECAY = 0.999
MODEL_SAVE_FREQUENCY = 50
BUFFER_SIZE = 1000
STATES_SIZE = 3

class Command(IntEnum):
    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2

@dataclass
class EvaluatedCommand:
    state: List[np.ndarray]
    command: Command
    next_state: List[np.ndarray]
    reward: float

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
        self.conv1 = nn.Conv2d(3, 6, (7,7), stride=(3,3))
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

    def create_target_scores(self, evaluated_commands: List[EvaluatedCommand], scores: torch.tensor, target_network) -> List[torch.tensor]:
        target_tensors: List[torch.tensor] = []
        for i, evaluated_command in enumerate(evaluated_commands):
            target_tensor: torch.tensor = torch.clone(scores[i])
            _, best_score_next_state, _ = target_network.predict(evaluated_command.next_state)
            target_tensor[int(evaluated_command.command)] = evaluated_command.reward + GAMMA * best_score_next_state
            target_tensors.append(target_tensor)
        return target_tensors

    def state_to_tensor(self, input_state: List[np.ndarray]) -> torch.tensor:
        greyscale_images: List[np.ndarray] = [cv2.cvtColor(state, cv2.COLOR_BGR2GRAY) for state in input_state]
        normalized: List[np.ndarray] = [greyscale_image / 255.0 for greyscale_image in greyscale_images]
        as_tensors: List[torch.tensor] = [torch.tensor(normal.astype(np.float32)) for normal in normalized]
        as_tensor: torch.tensor = torch.stack(as_tensors)
        return as_tensor

    def train_model(self, evaluated_commands: List[EvaluatedCommand], target_network):
        # Update Model
        input_states: List[torch.tensor] = [self.state_to_tensor(command.state) for command in evaluated_commands]

        # Train Model
        loss = ""
        scores = ""
        optimizer = optim.Adam(self.parameters(), lr=LEARNING_RATE)
        for i in range(1):
            scores = self(torch.stack(input_states))
            target_scores: List[torch.tensor] = self.create_target_scores(evaluated_commands, scores, target_network)
            criterion = nn.MSELoss()
            loss = criterion(scores, torch.stack(target_scores))
            loss.backward()
            optimizer.step()

        # Print Information
        print("Model Update:")
        print_scores(scores, evaluated_commands, target_network)
        print(f"Loss: {loss}")

    def predict(self, state: List[np.ndarray]) -> Tuple[Command, float, torch.tensor]:
        as_tensor: torch.tensor = self.state_to_tensor(state)
        as_tensor = as_tensor[None, ...]
        scores: torch.tensor = self(as_tensor)
        as_command: Command = Command(int(torch.argmax(scores)))
        score: float = float(torch.max(scores))
        return as_command, score, scores

    def set_weights(self, other_network):
        self.load_state_dict(other_network.state_dict())

    def save_model(self, filepath: Path):
        torch.save(self.state_dict(), filepath)

    def load_model(self, filepath: Path):
        self.load_state_dict(torch.load(filepath))


def print_scores(scores: torch.tensor, evaluated_commands: List[EvaluatedCommand], target_network: QNetwork):
    all_commands: List[Command] = [command for command in Command]
    for j in range(scores.shape[0]):
        _, best_score_next_state, _ = target_network.predict(evaluated_commands[j].next_state)
        print(f"{str(evaluated_commands[j].command)} performed: {round(evaluated_commands[j].reward + GAMMA * best_score_next_state, 2)}")
        for i in range(len(all_commands)):
            print(f"{str(all_commands[i]).split('.')[1]}: {round(float(scores[j][i]), 2)}", end=" ")
        print("")
        print("")

# Init Car Racing
car_racing: CarRacing = gym.make('CarRacing-v1')
car_racing.reset()
all_commands: List[Command] = [command for command in Command]

# Init Network
weights_path: Path = Path() / "weights"
q_learner: QNetwork = QNetwork()
e: int = 0
epsilon: float = 1.0 * pow(EPSILON_DECAY, e * (50 / STEP_SIZE))
if len(sys.argv) > 1:
    e = int(sys.argv[1])
    weights_file: Path = weights_path / f"weights_{e}.pt"
    q_learner.load_model(weights_file)
    if len(sys.argv) > 2:
        epsilon = float(sys.argv[2])
q_target_net: QNetwork = QNetwork()
evaluated_commands: List[EvaluatedCommand] = []

while True:
    # Start Track Again
    car_racing.reset()
    if e % TARGET_NETWORK_UPDATE_FREQUENCY == 0:
        print(f"Episode: {e}")
        q_target_net.set_weights(q_learner)
    if e % MODEL_SAVE_FREQUENCY == 0 and e > 0:
        q_learner.save_model(weights_path / f"weights_{e}.pt")

    # Drive on Track until leaving Track
    negative_rewards_in_a_row: int = 0
    current_time: int = 1
    scores: torch.tensor = torch.zeros(10)
    commands: List[EvaluatedCommand] = []
    states: List[np.ndarray] = [car_racing.state for _ in range(STATES_SIZE)]
    while negative_rewards_in_a_row < (20 / STEP_SIZE) or current_time < 50:
        # Make Decision Where to Drive
        command, score, scores = q_learner.predict(states)
        performed_command: Command = command
        if random.random() < epsilon:
            performed_command = random.choice(all_commands)
        as_action: np.ndarray = get_action(performed_command)
        end_state: np.ndarray = car_racing.state
        accumulated_reward: float = 0
        for _ in range(STEP_SIZE):
            end_state, reward, done, info = car_racing.step(as_action)
            accumulated_reward += reward
            current_time += 1
            car_racing.render(mode="human")
        states.append(end_state)
        evaluated_commands.append(EvaluatedCommand(states[:STATES_SIZE], performed_command, states[1:], accumulated_reward + 1))
        states = states[1:STATES_SIZE+1]
        negative_rewards_in_a_row = negative_rewards_in_a_row + 1 if accumulated_reward < 0 else 0

        # Train Model
        if len(evaluated_commands) > BATCH_SIZE:
            if len(evaluated_commands) > 5000:
                evaluated_commands = evaluated_commands[1:BUFFER_SIZE+1]
            sampled = random.sample(evaluated_commands, BATCH_SIZE)
            #q_learner.train_model(sampled, q_target_net)
            epsilon *= EPSILON_DECAY

    print(f"Epsilon: {epsilon}")
    e += 1





