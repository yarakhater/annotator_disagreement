#!/bin/bash
#SBATCH --partition=V100
#SBATCH --job-name=fr_new_ab_text
#SBATCH --output=fr_new_text_output.txt
#SBATCH --error=fr_new_text_error.txt
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00

set -x
srun python3 -u new_text_freezed.py
