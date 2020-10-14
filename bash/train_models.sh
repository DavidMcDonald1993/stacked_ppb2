#!/bin/bash

#SBATCH --job-name=TRAINMODELS
#SBATCH --output=TRAINMODELS_%A_%a.out
#SBATCH --error=TRAINMODELS_%A_%a.err
#SBATCH --array=0-4
#SBATCH --time=10-00:00:00
#SBATCH -c=10
#SBATCH --mem=20G

models=(nn+nb)
fps=(morg2 morg3 maccs circular rdk)

num_models=${#models[@]}
num_fps=${#fps[@]}

model_id=$((SLURM_ARRAY_TASK_ID / num_fps % num_models))
fps_id=$((SLURM_ARRAY_TASK_ID % num_fps))

model=${models[$model_id]}
fp=${fps[$fps_id]}

output_file=models/${fp}-${model}.pkl

if [ ! -f ${output_file} ]
then
    echo writing to $output_file

    module purge
    module load bluebear
    module load Anaconda3/2018.12

    # pip install -r requirements.txt
    # conda install --file spec-file.txt
    # conda env update -f env.yml

    # conda env create -f env.yml
    source activate ppb2_env
    # conda install -c rdkit rdkit libboost=1.65.1 -y
    # conda install -c openeye openeye-toolkits -y
    # conda install -c conda-forge swifter -y
    # pip install --user scikit-multilearn
    # conda env update --file env.yml 

    args=$(echo --model ${model}\
        --fp ${fp}\
        )
    python ppb2/train_models.py ${args}
fi