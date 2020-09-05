#!/bin/bash

env=$1
algo=$2
seed=$3
max_frames=$4
expert_priority_modifier=$5
tf_dir=$6
echo "$env $algo $seed $max_frames $expert_priority_decay $tf_dir"
module load cuda cudnn
source "$tf_dir/bin/activate"

python dqfd.py --agent=$algo --seed=$seed --env_id="$env-v4" --expert_file="human_$env-v4_5.pkl" --max_frames=$max_frames --expert_priority_modifier=$expert_priority_modifier

