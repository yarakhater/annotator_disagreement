#!/bin/bash
#SBATCH --partition=P100
#SBATCH --job-name=ab_annotators_freeze
#SBATCH --output=annotators_freeze_output.txt
#SBATCH --error=annotators_freeze_error.txt
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00

set -x
srun python3 -u annotators_freeze.py
