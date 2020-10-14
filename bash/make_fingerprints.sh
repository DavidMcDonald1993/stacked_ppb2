#!/bin/bash

#SBATCH --job-name=FPS
#SBATCH --output=FPS.out
#SBATCH --error=FPS.err
#SBATCH --time=10-00:00:00
#SBATCH -c 10
#SBATCH --mem=5G

module purge
module load bluebear
module load Anaconda3/2018.12

source activate ppb2_env

python ppb2/get_fingerprints.py