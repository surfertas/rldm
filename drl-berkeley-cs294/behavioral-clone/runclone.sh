#!/bin/bash
# Runs behave_clone.py on specified rollouts.

python behave_clone.py ./rollouts/rollout-Humanoid-v1-5 Humanoid-v1 5 True
python behave_clone.py ./rollouts/rollout-Humanoid-v1-10 Humanoid-v1 10 True
python behave_clone.py ./rollouts/rollout-Humanoid-v1-20 Humanoid-v1 20 True
python behave_clone.py ./rollouts/rollout-Humanoid-v1-40 Humanoid-v1 40 True
python behave_clone.py ./rollouts/rollout-Humanoid-v1-100 Humanoid-v1 100 True

