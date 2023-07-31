#!/bin/bash
#SBATCH --partition=V100
#SBATCH --job-name=new_ab_text
#SBATCH --output=new_text_output.txt
#SBATCH --error=new_text_error.txt
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00

set -x
srun python3 -u new_text.py
