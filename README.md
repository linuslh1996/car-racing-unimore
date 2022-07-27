In the project, we implemented several approaches to solve the car racing problem.

# Installation

1. First, `swig` needs to be installed. Depending on the used package manager, type either `conda install -c anaconda swig`, `brew install swig` or `sudo apt install swig`
2. Go into the project directory and install all requirements with: `
````
pip install -r requirements.txt
````


# Q Learning

To train: 
````
python train_qlearning.py save_folder_name -t 0
````

To view the results:
````
python train_qlearning.py q_learning_trained -i 1500
````

# PPO

To train: 
````
python train_ppo.py save_folder_name -t 0
````

To view the results:
````
python train_ppo.py ppo_trained -i 1500
````

# Ruin and Recreate

To train: 
````
python train_ruin_and_recreate.py save_file_name -t
````

To view the results:
````
python train_ruin_and_recreate.py ruin_recreate.json -i
````
