#!/bin/bash

#SBATCH --job-name=STACK
#SBATCH --output=STACK_%A_%a.out
#SBATCH --error=STACK_%A_%a.err
#SBATCH --array=0-5
#SBATCH --time=10-00:00:00
#SBATCH -c 20
#SBATCH --mem=50G

N_PROC=20

splits=(split_0 split_1 split_2 split_3 split_4 complete)
models=("stack morg3-lr morg2-svc morg3-svc circular-svc rdk-svc")

num_splits=${#splits[@]}
num_models=${#models[@]}

split_id=$((SLURM_ARRAY_TASK_ID / num_models % num_splits))
model_id=$((SLURM_ARRAY_TASK_ID % num_models))

split=${splits[${split_id}]}
model=${models[${model_id}]}

compounds=splits/${split}/train.smi 
targets=splits/${split}/train.npz 

output_dir=models/${split}

# determine output file
model_arr=(${model})
output_file="${output_dir}/${model_arr[0]}-(${model_arr[1]}"
for model_name in ${model_arr[@]:2};
do 
    output_file=${output_file}"&"${model_name}
done
output_file=${output_file}").pkl"

if [ ! -f ${output_file} ]
then
    echo writing to ${output_file}

    module purge
    module load bluebear
    module load Anaconda3/2018.12

    source activate ppb2_env

    args=$(echo --compounds ${compounds}\
        --targets ${targets}\
        --model ${model}\
        --path ${output_dir} \
        --n_proc ${N_PROC} \
        )
    # echo $args
    ulimit -c 0
    python ppb2/train_model.py ${args}
fi