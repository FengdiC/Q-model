#!/bin/bash
module load cuda cudnn
source ../gail/tf/bin/activate
python3 dist_DQN.py --checkpoint_index=-1

