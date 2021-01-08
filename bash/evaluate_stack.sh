#!/bin/bash

#SBATCH --job-name=EVALSTACK
#SBATCH --output=EVALSTACK_%A_%a.out
#SBATCH --error=EVALSTACK_%A_%a.err
#SBATCH --array=0-1
#SBATCH --time=10-00:00:00
#SBATCH -c 8
#SBATCH --mem=50G

N_PROC=4

models=("stack morg2-ada morg3-ada rdk-ada" "stack morg2-nb morg3-nb rdk-nb")

num_models=${#models[@]}

model_id=$((SLURM_ARRAY_TASK_ID % num_models))
model=${models[$model_id]}

model_arr=(${model})
complete_model_name="${model_arr[0]}-(${model_arr[1]}"
for model_name in ${model_arr[@]:2};
do 
    complete_model_name=${complete_model_name}"&"${model_name}
done
complete_model_name=${complete_model_name}")"
output_file=results/${complete_model_name}-results.pkl

if [ ! -f ${output_file} ]
then
    echo writing to $output_file

    module purge
    module load bluebear
    module load Anaconda3/2018.12

    source activate ppb2_env

    args=$(echo --model ${model} --n_proc ${N_PROC})

    ulimit -c 0

    python ppb2/evaluate_models.py ${args}
fi