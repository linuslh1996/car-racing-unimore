import random
from pathlib import Path
from typing import List, Tuple
import json

from src import car_racing as cr

NUMBER_OF_COMMANDS = 10
NUMBER_OF_DESTROYED = 5
NUMBER_OF_NEIGHBOORS = 5
SECTION_SIZE = 5
NUMBER_STEPS = 4
STEPS_CUTOFF = 3


def create_destroyed_solution(command: List[cr.Command]) -> List[cr.Command]:
    neighbour: List[cr.Command] = []
    for i, single_command in enumerate(command):
        section_start: int = random.randint(0, NUMBER_OF_COMMANDS - SECTION_SIZE)
        section = range(section_start, section_start + SECTION_SIZE)
        if i in section:
            neighbour.append(random.choice(cr.Command.all_commands()))
        else:
            neighbour.append(single_command)
    return neighbour

def create_neighbour(command: List[cr.Command]) -> List[cr.Command]:
    neighbour: List[cr.Command] = []
    for i, single_command in enumerate(command):
        section_start: int = random.randint(0, NUMBER_OF_COMMANDS - SECTION_SIZE)
        section = range(section_start, section_start + SECTION_SIZE)
        if i in section and random.random() < 0.5:
            neighbour.append(random.choice(cr.Command.all_commands()))
        else:
            neighbour.append(single_command)
    return neighbour

def local_search(candidates: List[List[cr.Command]], already_performed_commands: List[cr.Command], car_racing: cr.CustomRacing) -> Tuple[List[cr.Command], float]:
    max_reward = 0
    max_command = candidates[0]
    for candidate in candidates:
        car_racing.reset(seed=0)
        accumulated_reward = 0
        for command in already_performed_commands + candidate:
            reward = car_racing.perform_step(command)
            accumulated_reward += reward
        if accumulated_reward > max_reward:
            max_reward = accumulated_reward
            max_command = candidate
    return max_command, max_reward

def load_commands(path: Path) -> List[cr.Command]:
    with path.open("r") as file:
        commands = json.load(file)
    return [cr.Command(command) for command in commands]

def save_commands(path: Path, commands: List[cr.Command]):
    with path.open("w") as file:
        json.dump(commands, file)

def perform_ruin_and_recreate(car_racing: cr.CustomRacing, metaheuristics_safe: Path, train: bool):
    car_racing.reset(seed=0)

    # Display Best Track so Far
    if not train:
        already_performed_commands: List[cr.Command] = load_commands(metaheuristics_safe) if metaheuristics_safe.exists() else []
        for command in already_performed_commands:
            car_racing.perform_step(command, render=True)
            if car_racing.done():
                print("Finished Track")
                return
        return

    # Train
    while not car_racing.done():

        # Initialize Everything
        already_performed_commands: List[cr.Command] = load_commands(metaheuristics_safe) if metaheuristics_safe.exists() else []
        initial_solution: List[cr.Command] = [random.choice(cr.Command.all_commands()) for i in range(NUMBER_OF_COMMANDS)]
        candidates_for_local_search: List[List[cr.Command]] = [create_neighbour(initial_solution) for i in range(NUMBER_OF_NEIGHBOORS)]
        candidates_for_local_search.append(initial_solution)
        best_solution: List[cr.Command] = initial_solution
        best_score: float = 0
        for step in range(NUMBER_STEPS):

            # Local Search, Destroy Parts of Solution, Local Search Again
            local_searched_solution, _ = local_search(candidates_for_local_search, already_performed_commands, car_racing)
            destroyed_solutions: List[List[cr.Command]] = [create_destroyed_solution(local_searched_solution) for i in range(NUMBER_OF_DESTROYED)]
            for destroyed_solution in destroyed_solutions:
                candidates_destroyed_of_local_search: List[List[cr.Command]] = [create_neighbour(destroyed_solution) for i in range(NUMBER_OF_NEIGHBOORS)]
                candidates_destroyed_of_local_search.append(destroyed_solution)
                best_neighbour_of_destroyed, score = local_search(candidates_destroyed_of_local_search, already_performed_commands, car_racing)
                if score > best_score:
                    best_solution = best_neighbour_of_destroyed
                    best_score = score

            # Print Result
            car_racing.reset(seed=0)
            for command in already_performed_commands + best_solution:
                car_racing.perform_step(command, render=True)
            print(f"Max Reward: {best_score}: {best_solution}")

            # Save
            save_commands(metaheuristics_safe, already_performed_commands + best_solution[:-STEPS_CUTOFF])


def get_metaheuristics_data(car_racing: cr.CustomRacing, metaheuristics_safe: Path) -> List[float]:
    commands: List[cr.Command] = load_commands(metaheuristics_safe)
    accumulated_reward: List[float] = []
    car_racing.reset(seed=0)
    for command in commands:
        car_racing.perform_step(command)
        accumulated_reward.append(car_racing.accumulated())
    return accumulated_reward

