#!/bin/bash

env=$1
algo=$2
seed=$3
max_frames=$4
expert_priority_modifier=$5
tf_dir=$6
lr_rate=$7
custom_id=$8
eps_max_frame=$9

echo "$env $algo $seed $max_frames $expert_priority_decay $tf_dir $lr_rate $custom_id $delete_expert"
module load cuda cudnn
source "$tf_dir/bin/activate"

python dqfd.py --agent=$algo --seed=$seed --env_id="$env-v4" --expert_file="dodging_human_$env-v4_2.pkl" --max_frames=$max_frames --expert_priority_modifier=$expert_priority_modifier --lr=$lr_rate --custom_id=$custom_id --eps_max_frame=$eps_max_frame


