#!/bin/bash
python_script=$1
env_id=$2
checkpoint_index=$3
seed=$4
expert_file=$5
num_sampled=$6
special=$7
bc_lr=$8
max_ent_coef=$9
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
echo $python_script $env_id $checkpoint_index $seed $expert_file $num_sampled $special $bc_lr $max_ent_coef

module load cuda cudnn
source ../tensorflow/bin/activate
python3 $python_script --env_id=$env_id --checkpoint_index=$checkpoint_index --seed=$seed --expert_file=$expert_file --num_sampled=$num_sampled --special_tag=$special --lr_bc=$bc_lr --max_ent_coef_bc=$max_ent_coef


