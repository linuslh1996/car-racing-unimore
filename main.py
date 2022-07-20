import sys
from pathlib import Path

import gym
from gym.envs.box2d import CarRacing

from ppo import PPONetwork, perform_ppo_learning
from q_learning import TrainingParameters
from metaheuristics import perform_metaheuristic_learning

car_racing: CarRacing = gym.make('CarRacing-v2')
perform_metaheuristic_learning(car_racing)


