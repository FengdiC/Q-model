#!/bin/bash
env_id=$1
checkpoint_index=$2
seed=$3
expert_file=$4
num_sampled=$5
special=$6
bc_lr=$7
max_ent_coef=$8
if [ -z "$env_id$" ]
then
	env_id="BreakoutDeterministic-v4"
fi
if [ -z "$checkpoint_index" ]
then
	checkpoint_index=-1
fi
if [ -z "$seed" ]
then
	seed=0
fi
module load cuda cudnn
source ../tensorflow/bin/activate
python3 expert_DQN.py --env_id=$env_id --checkpoint_index=$checkpoint_index --seed=$seed --expert_file=$expert_file --num_sampled=$num_sampled --special_tag=$special --lr_bc=$bc_lr --max_ent_coef_bc=$max_ent_coef


