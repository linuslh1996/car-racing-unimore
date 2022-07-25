import random
import statistics
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

from car_racing import state_to_tensor, Command, EvaluatedCommand, CustomRacing
from q_learning import MODEL_SAVE_FREQUENCY, STATES_SIZE, TrainingParameters

BATCH_SIZE = 128
BUFFER_SIZE = 2000

class PPONetwork(nn.Module):

    def __init__(self, weights_path: Path):
        # Init Network
        super(PPONetwork, self).__init__()
        super().__init__()
        self.conv1 = nn.Conv2d(3, 6, (7, 7))
        self.conv2 = nn.Conv2d(6, 12, (4, 4))
        output_nodes = 12 * 21 * 21 # out_channels * height * width
        self.policy_head = nn.Sequential(nn.Linear(output_nodes, 216), nn.ReLU(), nn.Linear(216, len(Command.all_commands())), nn.Softplus())

        # Init Cache
        self.current_evaluated_commands = []
        self.current_loss: torch.Tensor = torch.tensor(0)
        self.current_policy_scores: torch.Tensor = torch.tensor(0)
        self.current_rewards: torch.Tensor = torch.tensor(0)
        self.current_advantage_scores: torch.Tensor = torch.tensor(0)
        self.weights_path: Path = weights_path

    def forward(self, x):
        pool: nn.MaxPool2d = nn.MaxPool2d(2, 2)

        x = pool(fn.relu(self.conv1(x)))
        x = pool(fn.relu(self.conv2(x)))
        x = torch.flatten(x, 1)

        policy_probabilities: torch.Tensor = self.policy_head(x)
        return policy_probabilities

    def weights_valid(self) -> bool:
        return not torch.any(torch.isnan(self.conv1.weight))

    def save_model(self, episode: int):
        torch.save(self.state_dict(), self.weights_path / f"weights_{episode}.pt")

    def load_model(self, episode: int):
        self.load_state_dict(torch.load(self.weights_path / f"weights_{episode}.pt"))

    def update_weights(self, other_network):
        self.load_state_dict(other_network.state_dict())

    def predict(self, input_state: List[np.ndarray], in_train_mode: bool) -> Tuple[Command, float]:
        policy_scores: torch.Tensor = self.get_policy_scores([input_state])
        distribution: Categorical = Categorical(policy_scores)
        sampled_action = distribution.sample()
        command_to_take: Command = Command(0)
        if in_train_mode:
            command_to_take = Command(int(sampled_action))
        else:
            command_to_take = Command(int(torch.argmax(policy_scores)))
        log_probability = distribution.log_prob(sampled_action)
        return command_to_take, log_probability

    def advantage_scores(self, evaluated_commands: List[EvaluatedCommand]) -> torch.Tensor:
        next_steps_mean: torch.Tensor = torch.tensor([statistics.mean(command.rewards[1:5]) for command in evaluated_commands])
        advantages: torch.Tensor = next_steps_mean
        return advantages

    def get_policy_scores(self, states_to_predict: List[List[np.ndarray]]) -> torch.Tensor:
        states_to_predict: List[torch.Tensor] = [state_to_tensor(state) for state in states_to_predict]
        policy_scores = self(torch.stack(states_to_predict))
        sum_greater_zero: torch.Tensor = torch.stack([scores if torch.sum(scores) > 0
                                                  else torch.ones(len(Command.all_commands()))
                                                  for scores in policy_scores])
        return sum_greater_zero

    def train_model(self, evaluated_commands: List[EvaluatedCommand]):
        # Create Neccessary input tensors
        states_to_predict: List[List[np.ndarray]] = [command.state for command in evaluated_commands]
        performed_actions: torch.Tensor = torch.tensor([int(command.command) for command in evaluated_commands])
        old_probabilites: torch.Tensor = torch.tensor([command.log_probability for command in evaluated_commands])
        current_rewards: torch.Tensor = torch.tensor([command.rewards[0] for command in evaluated_commands])

        # Calculate Advantages
        policy_scores: torch.Tensor = self.get_policy_scores(states_to_predict)
        advantages: torch.Tensor = self.advantage_scores(evaluated_commands)

        # Calculate Ratio
        new_probabilities: Categorical = Categorical(policy_scores).log_prob(performed_actions)
        ratios: torch.Tensor = torch.exp(new_probabilities - old_probabilites)

        # Calculate Scores
        unclamped_score: torch.Tensor = ratios * advantages
        clamped_score: torch.Tensor = torch.clamp(ratios, 0.9, 1.1) * advantages
        action_loss = -torch.min(unclamped_score, clamped_score).mean()
        loss: torch.Tensor = action_loss
        optimizer = optim.Adam(self.parameters())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update Cache
        self.current_evaluated_commands = evaluated_commands
        self.current_loss = loss
        self.current_policy_scores = policy_scores
        self.current_advantage_scores = advantages
        self.current_rewards = current_rewards

    def print_information(self):
        all_commands: List[Command] = Command.all_commands()
        normalized_policy_scores = F.normalize(self.current_policy_scores, p=1)
        for j in range(len(self.current_evaluated_commands)):
            print(
                f"{str(self.current_evaluated_commands[j].command).split('.')[1]} performed: "
                f"{round(float(self.current_rewards[j]),2)}        ",
                end=" ")
            for i in range(len(all_commands)):
                print(f"{str(all_commands[i]).split('.')[1]}: {round(float(normalized_policy_scores[j][i]), 2)}", end=" ")
            print(" ")
            print(f"Advantage Score: {self.current_advantage_scores[j]}")
        print("")
        print(f"Loss: {round(float(self.current_loss), 2)}")


def perform_ppo_learning(car_racing: CustomRacing, ppo_network: PPONetwork, params: TrainingParameters):
    # Init Car Racing
    evaluated_commands: List[EvaluatedCommand] = []

    # Do Car Racing
    while car_racing.current_episode() <= 2500:
        car_racing.reset(seed=0)
        if car_racing.current_episode() % MODEL_SAVE_FREQUENCY == 0 and car_racing.current_episode() > 0:
            ppo_network.save_model(car_racing.current_episode())
        # Drive on Track until leaving Track
        states: List[np.ndarray] = [car_racing.current_state() for _ in range(STATES_SIZE)]
        episode_evaluated_commands: List[EvaluatedCommand] = []
        episode_rewards: List[float] = []

        while not car_racing.out_of_track():

            # Select Action
            command, log_prob = ppo_network.predict(states, params.train_model)
            accumulated_reward: float = car_racing.perform_step(command, render=not params.train_model)
            if car_racing.done():
                print("Finished Track!")
                ppo_network.save_model(car_racing.current_episode())
                break

            # Save Actions to Memory
            states.append(car_racing.current_state())
            episode_evaluated_commands.append(
                EvaluatedCommand(states[:STATES_SIZE], command, states[1:], [], log_prob))
            episode_rewards.append(accumulated_reward)
            states = states[1:STATES_SIZE + 1]

            # Train Model
            if len(evaluated_commands) > BATCH_SIZE and params.train_model:
                sampled = random.sample(evaluated_commands, BATCH_SIZE)
                ppo_network.save_model(-1)
                ppo_network.train_model(sampled)
                if not ppo_network.weights_valid():
                    ppo_network.load_model(-1)

        # Print Information
        if len(evaluated_commands) > BATCH_SIZE and params.train_model:
            ppo_network.print_information()
            print(f"Episode: {car_racing.current_episode()}")

        # For every Command, save the Reward for the 5 next steps after it
        for i in range(len(episode_evaluated_commands)):
            for j in range(5):
                reward = episode_rewards[i+j] if (i+j < len(episode_evaluated_commands) and episode_rewards[i+j] > 0) \
                                              else -15
                episode_evaluated_commands[i].rewards.append(reward)
        evaluated_commands += episode_evaluated_commands
        if len(evaluated_commands) > BUFFER_SIZE:
           evaluated_commands = evaluated_commands[-BUFFER_SIZE:]

        # Print for Inference Mode
        if not params.train_model:
            for i, episode in enumerate(episode_evaluated_commands):
                print(f"Move: {str(episode.command)}, Advantage Score: {ppo_network.advantage_scores(episode_evaluated_commands)[i]}")
            print("")




