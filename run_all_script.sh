#!/bin/bash
special=$1
python_script=$2
if [ -z "$special" ]
then
	echo "No special Arguments"
	exit 1
fi
if [ -z "$python_script" ]
then
	echo "No script Arguments"
	exit 1
fi

declare -a Environments=("BreakoutDeterministic-v4" "SeaquestDeterministic-v4" )
declare -a Expert=(5 150 )
declare -a bc_lr=(0.001 0.0002 0.00005 )
declare -a max_ent_coef=(0.5 1.0 2.0)

# Iterate the string array using for loop
for val in ${Environments[@]}; do
	for exp in ${Expert[@]}; do
	    for lr in ${bc_lr[@]}; do
	        for coef in ${max_ent_coef[@]}; do
                echo $val $exp $lr $coef
                sbatch --gres=gpu:1 --cpus-per-task=2 --account=rrg-dpmeger --mem=64G --time=2:00:00 ./run_dqn.sh $python_script $val -1 0 "expert_data.pkl" $exp $special $lr $coef
            done
        done
	done
done
