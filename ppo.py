import random
from pathlib import Path
from typing import List, Tuple

import gym
import numpy as np
import torch
from gym.envs.box2d import CarRacing
from torch import nn, optim
import torch.nn.functional as F

import torch.nn.functional as fn
from torch.distributions import Categorical

from car_racing import state_to_tensor, Command, EvaluatedCommand
from q_learning import MODEL_SAVE_FREQUENCY, STATES_SIZE, TrainingParameters

BATCH_SIZE = 128
BUFFER_SIZE = 1000

class PPONetwork(nn.Module):

    def __init__(self, weights_path: Path):
        # Init Network
        super(PPONetwork, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, (7, 7))
        self.conv2 = nn.Conv2d(6, 12, (4, 4))
        output_nodes = 12 * 21 * 21 # out_channels * height * width
        self.value_head = nn.Sequential(nn.Linear(output_nodes, 216), nn.ReLU(), nn.Linear(216, 1), nn.Softplus())
        self.policy_head = nn.Sequential(nn.Linear(output_nodes, 216), nn.ReLU(), nn.Linear(216, len(Command.all_commands())), nn.Softplus())

        # Init Cache
        self.current_evaluated_commands = []
        self.current_loss: torch.Tensor = torch.tensor(0)
        self.current_policy_scores: torch.Tensor = torch.tensor(0)
        self.current_state_scores: torch.Tensor = torch.tensor(0)
        self.current_next_state_scores: torch.Tensor = torch.tensor(0)
        self.weights_path: Path = weights_path

    def forward(self, x):
        pool: nn.MaxPool2d = nn.MaxPool2d(2, 2)
        x = pool(fn.relu(self.conv1(x)))
        x = pool(fn.relu(self.conv2(x)))
        x = torch.flatten(x, 1)

        policy_probabilities: torch.Tensor = self.policy_head(x)
        state_scores: torch.Tensor = self.value_head(x)
        return policy_probabilities, state_scores

    def save_model(self, episode: int):
        torch.save(self.state_dict(), self.weights_path / f"weights_{episode}.pt")

    def load_model(self, episode: int):
        self.load_state_dict(torch.load(self.weights_path / f"weights_{episode}.pt"))

    def update_weights(self, other_network):
        self.load_state_dict(other_network.state_dict())

    def predict(self, input_state: List[np.ndarray], in_train_mode: bool) -> Tuple[Command, float]:
        as_tensor: torch.Tensor = state_to_tensor(input_state)
        as_tensor = as_tensor[None, ...]
        action_probabilities: torch.Tensor = self(as_tensor)[0]
        distribution: Categorical = Categorical(action_probabilities)
        sampled_action = distribution.sample()
        command_to_take: Command = Command(0)
        if in_train_mode:
            command_to_take = Command(int(sampled_action))
        else:
            command_to_take = Command(int(torch.argmax(action_probabilities)))
        log_probability = distribution.log_prob(sampled_action)
        return command_to_take, log_probability

    def train_model(self, evaluated_commands: List[EvaluatedCommand], target_network):
        # Create Neccessary input tensors
        states_to_predict: List[torch.Tensor] = [state_to_tensor(command.state) for command in evaluated_commands]
        next_states: List[torch.Tensor] = [state_to_tensor(command.next_state) for command in evaluated_commands]
        performed_actions: torch.Tensor = torch.tensor([int(command.command) for command in evaluated_commands])
        old_probabilites: torch.Tensor = torch.tensor([command.log_probability for command in evaluated_commands])
        rewards: torch.Tensor = torch.tensor([command.reward for command in evaluated_commands])

        # Calculate Advantages
        policy_scores, state_scores = self(torch.stack(states_to_predict))
        next_state_scores = self(torch.stack(next_states))[1]
        frozen_next_state_scores = target_network(torch.stack(next_states))[1]
        state_scores, next_state_scores = state_scores.squeeze(), next_state_scores.squeeze()
        advantages: torch.Tensor = next_state_scores - state_scores

        # Calculate Ratio
        new_probabilities: Categorical = Categorical(policy_scores).log_prob(performed_actions)
        ratios: torch.Tensor = torch.exp(new_probabilities - old_probabilites)

        # Calculate Scores
        unclamped_score: torch.Tensor = ratios * advantages
        clamped_score: torch.Tensor = torch.clamp(ratios, 0.9, 1.1) * advantages
        action_loss = -torch.min(unclamped_score, clamped_score).mean()
        scores_target = next_state_scores + rewards
        value_loss = F.smooth_l1_loss(state_scores, scores_target)
        loss: torch.Tensor = action_loss + 2. * value_loss
        optimizer = optim.Adam(self.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Cache
        self.current_evaluated_commands = evaluated_commands
        self.current_loss = loss
        self.current_policy_scores = policy_scores
        self.current_state_scores = state_scores
        self.current_next_state_scores = next_state_scores

    def print_information(self):
        all_commands: List[Command] = Command.all_commands()
        for j in range(len(self.current_evaluated_commands)):
            print(
                f"{str(self.current_evaluated_commands[j].command).split('.')[1]} performed: "
                f"{round(self.current_evaluated_commands[j].reward,2)}        ",
                end=" ")
            for i in range(len(all_commands)):
                print(f"{str(all_commands[i]).split('.')[1]}: {round(float(self.current_policy_scores[j][i]), 2)}", end=" ")
            print(" ")
            print(f"Current State Score: {round(float(self.current_state_scores[j]), 2)}, Next State Score: "
                  f"{round(float(self.current_next_state_scores[j]), 2)}")
        print("")
        print(f"Loss: {round(float(self.current_loss), 2)}")


def perform_ppo_learning(start_episode: int, ppo_network: PPONetwork, params: TrainingParameters):
    # Init Car Racing
    car_racing: CarRacing = gym.make('CarRacing-v1')
    car_racing.reset()
    evaluated_commands: List[EvaluatedCommand] = []
    current_episode: int = start_episode
    target_network: PPONetwork = PPONetwork(Path())
    target_network.update_weights(ppo_network)

    # Do Car Racing
    while current_episode <= 1500:
        car_racing.reset()
        if current_episode % MODEL_SAVE_FREQUENCY == 0 and current_episode > 0:
            ppo_network.save_model(current_episode)
        if current_episode % 50 == 0:
            target_network.update_weights(ppo_network)

        # Drive on Track until leaving Track
        negative_rewards_in_a_row: int = 0
        current_time: int = 1
        states: List[np.ndarray] = [car_racing.state for _ in range(STATES_SIZE)]
        while negative_rewards_in_a_row < (20 / params.step_size) or current_time < 50:

            # Select Action and Perform it "STEP_SIZE" times
            command, log_prob = ppo_network.predict(states, params.train_model)
            as_action: np.ndarray = command.as_action()
            accumulated_reward: float = 0
            for _ in range(params.step_size):
                end_state, reward, done, info = car_racing.step(as_action)
                accumulated_reward += reward
                current_time += 1
                if not params.train_model:
                    car_racing.render(mode="human")

            # Save Actions to Memory
            states.append(car_racing.state)
            evaluated_commands.append(
                EvaluatedCommand(states[:STATES_SIZE], command, states[1:], accumulated_reward, log_prob))
            states = states[1:STATES_SIZE + 1]
            negative_rewards_in_a_row = negative_rewards_in_a_row + 1 if accumulated_reward < 0 else 0

            # Train Model
            if len(evaluated_commands) > BATCH_SIZE and params.train_model:
                if len(evaluated_commands) > BUFFER_SIZE:
                    evaluated_commands = evaluated_commands[1:BUFFER_SIZE + 1]
                sampled = random.sample(evaluated_commands, BATCH_SIZE)
                ppo_network.train_model(sampled, target_network)

        # Print Information
        current_episode += 1
        if len(evaluated_commands) > BATCH_SIZE and params.train_model:
            ppo_network.print_information()
            print(f"Episode: {current_episode}")



