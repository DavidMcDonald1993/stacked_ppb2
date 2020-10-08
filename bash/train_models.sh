#!/bin/bash

#SBATCH --job-name=TRAINMODELS
#SBATCH --output=TRAINMODELS_%A_%a.out
#SBATCH --error=TRAINMODELS_%A_%a.err
#SBATCH --array=0-1682
#SBATCH --time=10-00:00:00
#SBATCH --ntasks=1
#SBATCH --mem=10G

# models=(nn nb nb+nn)
models=(nb+nn)
# fps=(morg2 morg3 maccs circular rdk)
fps=(morg2)
target_ids=({0..1682})

num_models=${#models[@]}
num_fps=${#fps[@]}
num_targets=${#target_ids[@]}

model_id=$((SLURM_ARRAY_TASK_ID / (num_targets * num_fps) % num_models))
fps_id=$((SLURM_ARRAY_TASK_ID / num_targets % num_fps))
target_id=$((SLURM_ARRAY_TASK_ID % num_targets))

model=${models[$model_id]}
fp=${fps[$fps_id]}

output_file=models/target-${target_id}-${fp}-${model}.pkl

if [ ! -f ${output_file} ]
then
    echo writing to $output_file

    module purge
    module load bluebear
    module load Anaconda3/2018.12

    # pip install -r requirements.txt
    # conda install --file spec-file.txt
    conda create env -n ppb2_env -f env.yml
    conda activate ppb2_env

    args=$(echo --model ${model}\
        --fp ${fp}\
        --target_id ${target_id})
    python ppb2/train_models.py ${args}
fi