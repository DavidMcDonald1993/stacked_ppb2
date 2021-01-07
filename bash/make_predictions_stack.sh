#!/bin/bash

#SBATCH --job-name=PREDMODELS
#SBATCH --output=PREDMODELS_%A_%a.out
#SBATCH --error=PREDMODELS_%A_%a.err
#SBATCH --array=0-11
#SBATCH --time=10-00:00:00
#SBATCH -c 8
#SBATCH --mem=50G

N_PROC=4

splits=(split_0 split_1 split_2 split_3 split_4 complete)
models=("stack morg2-ada morg3-ada rdk-ada" "stack morg2-nb morg3-nb rdk-nb")

num_splits=${#splits[@]}
num_models=${#models[@]}

split_id=$((SLURM_ARRAY_TASK_ID / num_models % num_splits))
model_id=$((SLURM_ARRAY_TASK_ID % num_models))

split=${splits[${split_id}]}
model=${models[${model_id}]}

query=splits/${split}/test.smi 

model_arr=(${model})
model="${model_arr[0]}-(${model_arr[1]}"
for model_name in ${model_arr[@]:2};
do 
    model=${model}"&"${model_name}
done
model=${model}")"

model_dir=models/${split}
model_file=${model_dir}/${model}.pkl.gz

echo $model_file

output_dir=predictions/${split}
output_file=${output_dir}/${model}.pkl-test/probs.csv.gz

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