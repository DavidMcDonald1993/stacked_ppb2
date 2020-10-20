#!/bin/bash

#SBATCH --job-name=TRAINMODELS
#SBATCH --output=TRAINMODELS_%A_%a.out
#SBATCH --error=TRAINMODELS_%A_%a.err
#SBATCH --array=0-215
#SBATCH --time=10-00:00:00
#SBATCH -c 10
#SBATCH --mem=20G

splits=(split_0 split_1 split_2 split_3 split_4 complete)
models=(nn nb nn+nb svc lr bag)
fps=(morg2 morg3 maccs circular rdk rdk_maccs)

num_splits=${#splits[@]}
num_models=${#models[@]}
num_fps=${#fps[@]}

split_id=$((SLURM_ARRAY_TASK_ID / (num_fps * num_models) % num_splits))
model_id=$((SLURM_ARRAY_TASK_ID / num_fps % num_models))
fps_id=$((SLURM_ARRAY_TASK_ID % num_fps))

split=${splits[${split_id}]}
model=${models[${model_id}]}
fp=${fps[${fps_id}]}

compounds=splits/${split}/train.smi 
targets=splits/${split}/train.npz 

model=${fp}-${model}

output_dir=models/${split}
output_file=${output_dir}/${model}.pkl

if [ ! -f ${output_file} ]
then
    echo writing to $output_file

    module purge
    module load bluebear
    module load Anaconda3/2018.12

    source activate ppb2_env

    args=$(echo --compounds ${compounds}\
        --targets ${targets}\
        --model ${model}\
        --path ${output_dir} \
        )
    # echo $args
    ulimit -c 0

    python ppb2/train_model.py ${args}
fi