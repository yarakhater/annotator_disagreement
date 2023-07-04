#!/bin/bash
#SBATCH --partition=P100
#SBATCH --job-name=ab_text_freeze
#SBATCH --output=text_freeze_output.txt
#SBATCH --error=text_freeze_error.txt
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00

set -x
srun python3 -u text_freeze.py
