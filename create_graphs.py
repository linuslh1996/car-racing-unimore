import json

import seaborn as sns
import pandas as pd
from pandas import DataFrame
from pathlib import Path
import matplotlib.pyplot as plt

metadata_folder: Path = Path() / "data" / "metadata"
figures_folder: Path = Path() / "data" / "figures"
ppo_data: DataFrame = pd.read_csv(metadata_folder / "ppo.csv")
q_learning_data: DataFrame = pd.read_csv(metadata_folder / "q_learning.csv")
ruin_recreate_data: DataFrame = pd.read_csv(metadata_folder / "ruin_and_recreate.csv")

# Plot QLearning Plots
sns.lineplot(data=q_learning_data, x="episode", y="average_q_value")
plt.savefig(figures_folder / "average_q_value.png")
plt.clf()

average_rewards_plot = sns.lineplot(data=q_learning_data, x="episode", y="average_rewards")
average_rewards_plot.set(ylim=(0,500))
plt.savefig(figures_folder / "average_rewards_q_learning.png")
plt.clf()

# Plot PPO Plots
average_rewards_plot = sns.lineplot(data=ppo_data, x="episode", y="average_reward", )
average_rewards_plot.set(ylim=(0,500))
plt.savefig(figures_folder / "average_rewards_ppo.png")
plt.clf()

# Compare PPO and Metaheuristics
successful_completion = json.loads(ppo_data.iloc[29]["reward_development"])[9]
as_dataframe = DataFrame([{"step": i, "total_reward": successful_completion[i], "algorithm": "ppo"} for i in range(len(successful_completion))])
combined = pd.concat([as_dataframe,ruin_recreate_data], ignore_index=True)
sns.lineplot(data=combined, x="step", y="total_reward", hue="algorithm")
plt.savefig(figures_folder / "comparison_ppo_metaheuristics.png")