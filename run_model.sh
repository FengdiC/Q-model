#!/bin/bash

#env=$1
#algo=$2
#seed=$3
#max_frames=$4
#expert_priority_modifier=$5
#tf_dir=$6
#lr_rate=$7
#custom_id=$8
#delete_expert=$9
expert_dir = '/home/yutonyan/Q-model/'
agent = $1
seed = $2
decay = $3
power = $4

#echo "$env $algo $seed $max_frames $expert_priority_decay $tf_dir $lr_rate $custom_id $delete_expert"
echo "$expert_dir $agent $seed $decay $power"
module load cuda cudnn
source "/home/yuntonyan/tf/bin/activate"

#python dqfd.py --agent=$algo --seed=$seed --env_id="$env-v4" --expert_file="human_$env-v4_1.pkl" --max_frames=$max_frames --expert_priority_modifier=$expert_priority_modifier --lr=$lr_rate --custom_id=$custom_id --delete_expert=$delete_expert

python /home/yutonyan/Q-model/dqfd.py --agent=$agent --seed = $seed --expert_dir=$expert_dir --decay=$decay --power=$power
