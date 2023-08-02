#!/bin/bash -l 
#SBATCH --time=24:00:00
#SBATCH --ntasks=15
#SBATCH --mem=450gb
export PYTHONUNBUFFERED=1

module load python3
module load cuda/10.1

source activate torch1.8

python validate.py
