#!/bin/bash
module load cuda cudnn
source ../tensorflow/bin/activate
python3 dist_DQN.py --checkpoint_index=3008226 --initial_exploration=0.05

