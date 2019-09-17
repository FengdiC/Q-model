#!/bin/bash

declare -a Environments=("BreakoutDeterministic-v4" "AsterixDeterministic-v4" "QbertDeterministic-v4" "SeaquestDeterministic-v4" )
declare -a Expert=(1 5 30 )

# Iterate the string array using for loop
for val in ${Environments[@]}; do
	for exp in ${Expert[@]}; do
		echo $val $exp
		sbatch --gres=gpu:1 --cpus-per-task=2 --account=rrg-dpmeger --mem=16G --time=23:55:00 ./run_fengdi_expert_dqn.sh $val -1 0 "expert_data.pkl" $exp
	done
done
#sbatch --gres=gpu:1 --cpus-per-task=2 --account=rrg-dpmeger --mem=16G --time=23:55:00 
