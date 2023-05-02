import sys
from pathlib import Path

from pandas import DataFrame

from src.car_racing import CustomRacing
from src.ruin_and_recreate import perform_ruin_and_recreate, get_metaheuristics_data

# Initialization
solution_dir: Path = Path() / "data" / "results"
metadata_dir: Path = Path() / "data" / "metadata"
solution_path: Path = solution_dir / sys.argv[1]
solution_dir.mkdir(parents=True, exist_ok=True)
assert sys.argv[2] == "-t" or sys.argv[2] == "-i" or sys.argv[2] == "-d"
car_racing: CustomRacing = CustomRacing(0)

# Training Mode
if sys.argv[2] == "-t":
    perform_ruin_and_recreate(car_racing, solution_path, True)
    
# Inference Mode
elif sys.argv[2] == "-i":
    perform_ruin_and_recreate(car_racing, solution_path, False)
    
# Collect Metadata Mode (for plots)
elif sys.argv[2] == "-d":
    data = get_metaheuristics_data(car_racing, solution_path)
    as_dataframe = DataFrame([{"step": i, "total_reward": data[i], "algorithm": "ruin_and_recreate"} for i in range(len(data))])
    as_dataframe.to_csv(metadata_dir / "ruin_and_recreate.csv")



