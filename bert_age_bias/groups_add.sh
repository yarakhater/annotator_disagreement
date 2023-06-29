#!/bin/bash
#SBATCH --partition=V100
#SBATCH --job-name=groups_add
#SBATCH --output=groups_add_output.txt
#SBATCH --error=groups_add_error.txt
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00

set -x
srun python3 -u groups_add.py
