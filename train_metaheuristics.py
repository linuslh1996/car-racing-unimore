import sys
from pathlib import Path

from car_racing import CustomRacing
from ppo import PPONetwork, perform_ppo_learning
from metaheuristics import perform_ruin_and_recreate
from q_learning import TrainingParameters

solution_dir: Path = Path() / "metaheuristics_safe"
solution_path: Path = solution_dir / sys.argv[2]
solution_dir.mkdir(parents=True, exist_ok=True)
assert sys.argv[1] == "-t" or sys.argv[1] == "-i"
car_racing: CustomRacing = CustomRacing(0)
if sys.argv[1] == "-t":
    perform_ruin_and_recreate(car_racing, solution_path, True)
else:
    perform_ruin_and_recreate(car_racing, solution_path, False)


