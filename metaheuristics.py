import random
from typing import List

from gym.envs.box2d import CarRacing
import car_racing as cr


STEP_SIZE = 10
NUMBER_OF_COMMANDS = 25
NUMBER_OF_NEIGHBOURS = 20
SECTION_SIZE = 5


def create_neighbour(command: List[cr.Command]) -> List[cr.Command]:
    neighbour: List[cr.Command] = []
    for i, single_command in enumerate(command):
        section_start: int = random.randint(0, NUMBER_OF_COMMANDS - SECTION_SIZE)
        section = range(section_start, section_start + SECTION_SIZE)
        if random.random() < 0.5 and i in section:
            neighbour.append(random.choice(cr.Command.all_commands()))
        else:
            neighbour.append(single_command)
    return neighbour


def perform_metaheuristic_learning(car_racing: CarRacing):
    car_racing.reset(seed=0)
    cr.reset(car_racing)
    current_command: List[cr.Command] = [random.choice(cr.Command.all_commands()) for i in range(NUMBER_OF_COMMANDS)]
    max_reward = 0
    max_command = current_command
    for episode in range(20):
        # Perform Local Search
        neighbours: List[List[cr.Command]] = [create_neighbour(max_command) for i in range(NUMBER_OF_NEIGHBOURS)]
        for neighbour in neighbours:
            car_racing.reset(seed=0)
            accumulated_reward = 0
            for command in neighbour:
                for i in range(STEP_SIZE):
                    end_state, reward, finished_track, info = car_racing.step(command.as_action())
                    accumulated_reward += reward
            if accumulated_reward > max_reward:
                max_reward = accumulated_reward
                max_command = neighbour

        # Print Result
        car_racing.reset(seed=0)
        for command in max_command:
            for i in range(STEP_SIZE):
                car_racing.render(mode="human")
                end_state, reward, finished_track, info = car_racing.step(command.as_action())
        print(f"Max Reward: {max_reward}: {max_command}")

