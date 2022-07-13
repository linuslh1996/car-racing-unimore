import random
from dataclasses import dataclass
from pathlib import Path
from typing import List

import gym
from gym.envs.box2d import CarRacing
import numpy as np
import sys
from network import EvaluatedCommand, Command, QNetwork

BATCH_SIZE: int = 64
EPSILON_DECAY = 0.999
MODEL_SAVE_FREQUENCY = 50
BUFFER_SIZE = 1000
STATES_SIZE = 3
GAMMA = 0.95
LEARNING_RATE_DECAY = 1.0

@dataclass
class TrainingParameters:
    step_size: int
    learning_rate: float
    target_network_update_frequency: int
    train_model: bool

def learn_q_values(start_episode: int, start_epsilon: float, q_learner: QNetwork, params: TrainingParameters):
    # Init Car Racing
    car_racing: CarRacing = gym.make('CarRacing-v1')
    car_racing.reset()
    all_commands: List[Command] = [command for command in Command]
    q_target_net: QNetwork = QNetwork(Path())
    q_target_net.set_weights(q_learner)
    evaluated_commands: List[EvaluatedCommand] = []
    current_episode: int = start_episode
    epsilon: float = start_epsilon

    # Do Car Racing
    while current_episode <= 1000:
        car_racing.reset()
        if current_episode % params.target_network_update_frequency == 0:
            q_target_net.set_weights(q_learner)
        if current_episode % MODEL_SAVE_FREQUENCY == 0 and current_episode > 0:
            q_learner.save_model(current_episode)
        if current_episode == 50:
            params.learning_rate *= LEARNING_RATE_DECAY

        # Drive on Track until leaving Track
        negative_rewards_in_a_row: int = 0
        current_time: int = 1
        states: List[np.ndarray] = [car_racing.state for _ in range(STATES_SIZE)]
        while negative_rewards_in_a_row < (20 / params.step_size) or current_time < 50:

            # Select Action and Perform it "STEP_SIZE" times
            command, score, scores = q_learner.predict(states)
            performed_command: Command = command
            if random.random() < epsilon:
                performed_command = random.choice(all_commands)
            as_action: np.ndarray = performed_command.as_action()
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
                EvaluatedCommand(states[:STATES_SIZE], performed_command, states[1:], accumulated_reward + 1))
            states = states[1:STATES_SIZE + 1]
            negative_rewards_in_a_row = negative_rewards_in_a_row + 1 if accumulated_reward < 0 else 0

            # Train Model
            if len(evaluated_commands) > BATCH_SIZE and params.train_model:
                if len(evaluated_commands) > 5000:
                    evaluated_commands = evaluated_commands[1:BUFFER_SIZE + 1]
                sampled = random.sample(evaluated_commands, BATCH_SIZE)
                q_learner.train_model(sampled, params.learning_rate, GAMMA, q_target_net)
                epsilon *= EPSILON_DECAY
        
        # Print Information
        if len(evaluated_commands) > BATCH_SIZE and params.train_model:
            q_learner.print_scores(q_learner.current_score, q_learner.current_evaluated_commands, q_learner.current_loss, GAMMA, q_target_net)
            print(f"Epsilon: {epsilon}")
            print(f"Episode: {current_episode}")
            current_episode += 1


# Init Network
weights_path: Path = Path() / "weights"

# Load Model and Continue Training
if len(sys.argv) > 1:
    assert sys.argv[1] == "-t" or sys.argv[1] == "-i"
    in_training = True if sys.argv[1] == "-t" else False
    weights_folder: Path = weights_path / sys.argv[2]
    weights_folder.mkdir(parents=True, exist_ok=True)
    start_episode: int = int(sys.argv[3])
    q_learner: QNetwork = QNetwork(weights_folder)
    if start_episode > 0:
        q_learner.load_model(start_episode)
    epsilon: float = 1.0
    if len(sys.argv) > 4:
        epsilon = float(sys.argv[4])
    parameters: List[str] = sys.argv[2].split("_")
    step_size, learning_rate, target_update = int(parameters[0]), float(parameters[1]), int(parameters[2])
    learn_q_values(start_episode, epsilon, q_learner, TrainingParameters(step_size, learning_rate, target_update, in_training))


# Train Model with multiple combinations
else:
    combinations: List[TrainingParameters] = [
        TrainingParameters(10, 0.001, 50, True),
        TrainingParameters(5, 0.001, 25, True),
        TrainingParameters(10, 0.0001, 50, True),
        TrainingParameters(10, 0.001, 25, True),
        TrainingParameters(10, 0.001, 100, True),
        TrainingParameters(20, 0.001, 50, True)
    ]

    # Try Combinations
    for param in combinations:
        file_path = weights_path / f"{param.step_size}_{param.learning_rate}_{param.target_network_update_frequency}"
        file_path.mkdir(parents=True, exist_ok=True)
        q_learner: QNetwork = QNetwork(file_path)
        learn_q_values(0, 1.0, q_learner, param)








