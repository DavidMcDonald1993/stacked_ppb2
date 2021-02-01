#!/bin/bash

#SBATCH --job-name=EVALMODELS
#SBATCH --output=EVALMODELS_%A_%a.out
#SBATCH --error=EVALMODELS_%A_%a.err
#SBATCH --array=0-41
#SBATCH --time=10-00:00:00
#SBATCH -c 10
#SBATCH --mem=20G

N_PROC=10

models=(dum nb nn nn+nb bag ada xgc)
fps=(morg2 morg3 maccs circular rdk rdk_maccs)

num_models=${#models[@]}
num_fps=${#fps[@]}

model_id=$((SLURM_ARRAY_TASK_ID / num_fps % num_models))
fps_id=$((SLURM_ARRAY_TASK_ID  % num_fps))

model=${models[$model_id]}
fp=${fps[$fps_id]}

output_file=results/${fp}-${model}-results.pkl

if [ ! -f ${output_file} ]
then
    echo writing to $output_file

    module purge
    module load bluebear
    module load Anaconda3/2018.12

    source activate ppb2_env

    args=$(echo --model ${fp}-${model} --n_proc ${N_PROC})

    ulimit -c 0

    python ppb2/evaluate_models.py ${args}
fi