#!/bin/bash
module load cuda cudnn
source ../gail/tf/bin/activate
python3 DQN.py --checkpoint_index=-1

