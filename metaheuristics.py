import random
from pathlib import Path
from typing import List
import json

from gym.envs.box2d import CarRacing
import car_racing as cr


NUMBER_OF_COMMANDS = 10
NUMBER_OF_NEIGHBOURS = 20
SECTION_SIZE = 5
NUMBER_STEPS = 10
STEPS_CUTOFF = 3


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

def load_commands(path: Path) -> List[cr.Command]:
    with path.open("r") as file:
        commands = json.load(file)
    return [cr.Command(command) for command in commands]


def save_commands(path: Path, commands: List[cr.Command]):
    with path.open("w") as file:
        json.dump(commands, file)

def perform_metaheuristic_learning(car_racing: cr.CustomRacing, metaheuristics_safe: Path, train: bool):
    car_racing.reset(seed=0)
    if not train:
        already_performed_commands: List[cr.Command] = load_commands(metaheuristics_safe) if metaheuristics_safe.exists() else []
        for command in already_performed_commands:
            car_racing.perform_step(command, render=True)
            if car_racing.done():
                print("Finished Track")
                return
        return

    while not car_racing.done():
        # Initialize Everything
        already_performed_commands: List[cr.Command] = load_commands(metaheuristics_safe) if metaheuristics_safe.exists() else []
        current_command: List[cr.Command] = [random.choice(cr.Command.all_commands()) for i in range(NUMBER_OF_COMMANDS)]
        max_reward = 0
        max_command = current_command
        for step in range(NUMBER_STEPS):

            # Perform Local Search
            neighbours: List[List[cr.Command]] = [create_neighbour(max_command) for i in range(NUMBER_OF_NEIGHBOURS)]
            for neighbour in neighbours:
                car_racing.reset(seed=0)
                accumulated_reward = 0
                for command in already_performed_commands + neighbour:
                    reward = car_racing.perform_step(command)
                    accumulated_reward += reward
                if accumulated_reward > max_reward:
                    max_reward = accumulated_reward
                    max_command = neighbour

            # Print Result
            car_racing.reset(seed=0)
            for command in already_performed_commands + max_command:
                car_racing.perform_step(command, render=True)
            print(f"Max Reward: {max_reward}: {max_command}")

            # Save
            save_commands(metaheuristics_safe, already_performed_commands + max_command[:-STEPS_CUTOFF])


