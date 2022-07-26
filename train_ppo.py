import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

from pandas import DataFrame

from src.car_racing import CustomRacing
from src.ppo import PPONetwork, perform_ppo_learning, get_ppo_data, PPOMetadata


weights_path: Path = Path() / "data" / "results"
weights_folder: Path = weights_path / sys.argv[1]
weights_folder.mkdir(parents=True, exist_ok=True)
metadata_path: Path = Path() / "data" / "metadata"
network = PPONetwork(weights_folder)
assert sys.argv[2] == "-t" or sys.argv[2] == "-i" or sys.argv[2] == "-d"
episode: int = int(sys.argv[3])
car_racing: CustomRacing = CustomRacing(episode)
if sys.argv[2] == "-t":
    if episode > 0:
        network.load_model(episode)
    perform_ppo_learning(car_racing, network, True)
elif sys.argv[2] == "-i":
    network.load_model(episode)
    perform_ppo_learning(car_racing, network, False)
elif sys.argv[2] == "-d":
    metadata: List[PPOMetadata] = get_ppo_data(car_racing, network)
    as_dataframe: DataFrame = DataFrame([asdict(data) for data in metadata])
    as_dataframe.to_csv(metadata_path / "ppo.csv")



