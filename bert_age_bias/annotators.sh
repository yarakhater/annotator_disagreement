#!/bin/bash
#SBATCH --partition=V100
#SBATCH --job-name=ab_annotators
#SBATCH --output=annotators_output.txt
#SBATCH --error=annotators_error.txt
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00

set -x
srun python3 -u annotators.py
