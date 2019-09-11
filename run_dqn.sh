#!/bin/bash
module load cuda cudnn
source ../tensorflow/bin/activate
python3 DQN.py --checkpoint_index=3007417 --initial_exploration=0.05

