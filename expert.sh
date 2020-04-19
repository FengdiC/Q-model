#!/bin/bash

source $HOME/.bashrc
conda activate base
module load python/3.6 cuda/9.0/cudnn/7.5
source $HOME/env/bin/activate
source activate base

python dqfd.py --agent='expert'

