#!/bin/bash
source ~/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
cd ~/IsaacLab

while true; do
    python scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Lift-Cube-UR5-Robotiq-v0 --max_iterations 30000 >> ~/training_lift_30k.txt 2>&1
    [ $? -eq 0 ] && break
    sleep 10
done
