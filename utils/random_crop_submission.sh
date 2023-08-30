#!/bin/bash
#SBATCH -J RANDOM_CROP
#SBATCH -p p-RTX2080
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --ntasks-per-node=6
#SBATCH -o out.%j
#SBATCH -e err.%j
#SBATCH --mem=6000
##################################################################

module load cuda11.3/toolkit/11.3.0
python utils/random_crop.py --datadir dataset/SLO_and_late_FFA/  --output_dir dataset/data_slolaffa/ 
