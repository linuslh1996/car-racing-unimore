import sys
from dataclasses import asdict
from pathlib import Path
from typing import List

from src.car_racing import CustomRacing
from src.q_learning import TrainingParameters, learn_q_values, QNetwork, create_graph_data, QLearningMetadata
from pandas import DataFrame

# Initialize
weights_path: Path = Path() / "data" / "results"
metadata_path: Path = Path() / "data" / "metadata"
weights_folder: Path = weights_path / sys.argv[1]
weights_folder.mkdir(parents=True, exist_ok=True)
network = QNetwork(weights_folder)
assert sys.argv[2] == "-t" or sys.argv[2] == "-i" or sys.argv[2] == "-d"
episode: int = int(sys.argv[3])
car_racing: CustomRacing = CustomRacing(episode)
    
# Training Mode
if sys.argv[2] == "-t":
    if episode > 0:
        network.load_model(episode)
    params: TrainingParameters = TrainingParameters(0.001, 50, True)
    learn_q_values(car_racing, 0.4, network, params)
    
# Inference Mode
elif sys.argv[2] == "-i":
    network.load_model(episode)
    params: TrainingParameters = TrainingParameters(0.001, 50, False)
    learn_q_values(car_racing, 0.0, network, params)
    
# Collect Metadata Mode (for plots)
elif sys.argv[2] == "-d":
    metadata: List[QLearningMetadata] = create_graph_data(car_racing, network)
    as_dataframe: DataFrame = DataFrame(data=[asdict(data) for data in metadata])
    as_dataframe.to_csv(metadata_path / "q_learning.csv")


