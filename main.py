import sys
from pathlib import Path
from ppo import PPONetwork, perform_ppo_learning
from q_learning import TrainingParameters

weights_path: Path = Path() / "weights"
weights_folder: Path = weights_path / "ppo_first_try_first_curve_works"
network = PPONetwork(weights_folder)
weights_folder.mkdir(parents=True, exist_ok=True)
assert sys.argv[1] == "-t" or sys.argv[1] == "-i"
episode: int = int(sys.argv[2])
if sys.argv[1] == "-t":
    if episode > 0:
        network.load_model(episode)
    params: TrainingParameters = TrainingParameters(10, 0.001, 50, True)
    perform_ppo_learning(episode, network, params)
else:
    network.load_model(episode)
    params: TrainingParameters = TrainingParameters(10, 0.001, 50, False)
    perform_ppo_learning(episode, network, params)


