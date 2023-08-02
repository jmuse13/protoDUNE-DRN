#!/bin/bash -l 
#SBATCH --time=24:00:00
#SBATCH --ntasks=30
#SBATCH --mem=450gb

source activate torch1.8

python make_input_files.py
