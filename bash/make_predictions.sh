#!/bin/bash

#SBATCH --job-name=PREDMODELS
#SBATCH --output=PREDMODELS_%A_%a.out
#SBATCH --error=PREDMODELS_%A_%a.err
#SBATCH --array=0-215
#SBATCH --time=10-00:00:00
#SBATCH -c 21
#SBATCH --mem=20G

N_PROC=8

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

query=splits/${split}/test.smi 

model=${fp}-${model}

model_dir=models/${split}
model_file=${model_dir}/${model}.pkl.gz

output_dir=predictions/${split}
output_file=${output_dir}/${model}-test/probs.csv

if [ ! -f ${output_file} ]
then
    echo writing to $output_file

    module purge
    module load bluebear
    module load Anaconda3/2018.12

    source activate ppb2_env

    args=$(echo --query ${query}\
        --model ${model_file}\
        --output ${output_dir} \
        --n_proc ${N_PROC}\
        )
    # echo $args
    ulimit -c 0

    python ppb2/make_prediction.py ${args}
fi