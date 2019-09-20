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

#declare -a Environments=("BreakoutDeterministic-v4" "SeaquestDeterministic-v4" "MontezumaRevengeDeterministic-v4" "MsPacman-Deterministic-v4" "QbertDeterministic-v4" )
#declare -a Environments=(SeaquestDeterministic-v4" "MsPacman-Deterministic-v4" "QbertDeterministic-v4" )
#declare -a learning_rates=(0.001 0.0005 0.00025 0.0001 0.00005 0.000025 0.00001)

declare -a Environments=("BreakoutDeterministic-v4" )
declare -a learning_rates=(0.001 )
# Iterate the string array using for loop
for val in ${Environments[@]}; do
    for lr in ${learning_rates[@]}; do
        echo $val $lr
        sbatch --gres=gpu:1 --cpus-per-task=2 --account=rrg-dpmeger --mem=64G --time=2:55:00 ./run_script.sh $python_script $val -1 0 "expert_data.pkl" 1 "$special_lr_$lr" $lr $coef
    done
done
