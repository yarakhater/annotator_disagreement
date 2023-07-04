#!/bin/bash
#SBATCH --partition=P100
#SBATCH --job-name=ab_groups_freeze
#SBATCH --output=groups_freeze_output.txt
#SBATCH --error=groups_freeze_error.txt
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00

set -x
srun python3 -u groups_freeze.py
