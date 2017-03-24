#!/bin/bash
# Runs run_expert.py on specified rollouts.


python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --num_rollouts 5
python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --num_rollouts 10
python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --num_rollouts 20
python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --num_rollouts 40
python run_expert.py experts/Humanoid-v1.pkl Humanoid-v1 --num_rollouts 100



