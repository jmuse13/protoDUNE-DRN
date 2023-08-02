#!/bin/bash -l 
#SBATCH --time=24:00:00
#SBATCH --ntasks=24
#SBATCH --mem=360gb
#SBATCH -p v100                                            
#SBATCH --gres=gpu:v100:2
export PYTHONUNBUFFERED=1

module load python3
module load cuda/10.1

source activate torch1.8

python train.py
