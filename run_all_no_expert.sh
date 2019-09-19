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

declare -a Environments=("BreakoutDeterministic-v4" "SeaquestDeterministic-v4" "MontezumaRevengeDeterministic-v4" "MsPacman-Deterministic-v4" "QbertDeterministic-v4" )

# Iterate the string array using for loop
for val in ${Environments[@]}; do
    echo $val
    sbatch --gres=gpu:1 --cpus-per-task=2 --account=rrg-dpmeger --mem=64G --time=23:55:00 ./run_script.sh $python_script $val -1 0 "expert_data.pkl" $exp $special $lr $coef
done
