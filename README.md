This is a seminar project that we did during our Erasmus stay at Unimore, Modena. In the project, we implemented several approaches to solve the car racing problem. To sum it up, Q Learning performed quite dissappointingly (even after we spend a lot of time parameter tuning!), PPO was surprisingly good and stable, and the EA approach worked fine but was very time-consuming.


# Installation

1. First, `swig` needs to be installed. Depending on the used package manager, type either `conda install -c anaconda swig`, `brew install swig` or `sudo apt install swig`
2. Go into the project directory and install all requirements with: `
````
pip install -r requirements.txt
````


# Q Learning

QLearning was very sensitive for us in regards to its parameters. This is in line with what the literature says, for example [here](https://w3.cs.jmu.edu/spragunr/papers/rldm2015.pdf). To solve this sensitivity, we could have implemented a more elaborate approach, for example [Double Q Learning]((https://arxiv.org/pdf/1509.06461.pdf). However, we decided against that, which means that our results for QLearning were a bit mediocre.

https://user-images.githubusercontent.com/33374886/235738296-69c9976c-c158-42ab-84d1-c6e017f01817.mp4


To train: 
````
python train_qlearning.py save_folder_name -t 0
````

To view the results:
````
python train_qlearning.py q_learning_trained -i 1500
````



# PPO

For Proximal Policy Optimization, we observed very good results. With an increasing amount of episodes, the total rewards were increasing as well. After training for 1500 episodes, we reached our target. Despite some oscillations, the reward improves continuously, until it reaches
500 after the 1500th episode. We observed that with our chosen advantage function, sometimes the car takes a small shortcut instead of entering the curve. We decided to allow this behaviour, as long as the car returns to the track quickly. This could be fixed by selecting an advantage function that penalises leaving the track more heavily. 



https://user-images.githubusercontent.com/33374886/235740405-ea5529ec-ceba-4dc3-8c31-b5d08dfea6d6.mp4


To train: 
````
python train_ppo.py save_folder_name -t 0
````

To view the results:
````
python train_ppo.py ppo_trained -i 1500
````

# Ruin and Recreate
This is an Evolutionary Algorithm approach. Basically, we perform random actions (DRIVE STRAIGHT, TURN LEFT, TURN RIGHT) and evaluate them. Then we modify the random sequences that performed best again at random points. Then again, again, etc. Even though the car is able to reach the goal reward, it's driving behaviour is far from ideal. When looking at the tracks, it is apparent that with PPO, the car often uses gas and drives straight. In our Ruin and Recreate approach, it usually drives a bit like a drunk person. This is because when the algorithm finds an okay sequence, it is highly likely to be stuck in a local optimum. To escape the local optima, we would need to run this algorithm with a lot more steps.



https://user-images.githubusercontent.com/33374886/235740638-ff5c3b62-82b1-4efb-85c1-7bba8104613b.mp4


To train: 
````
python train_ruin_and_recreate.py save_file_name -t
````

To view the results:
````
python train_ruin_and_recreate.py ruin_recreate.json -i
````
