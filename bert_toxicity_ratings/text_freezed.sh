#!/bin/bash
#SBATCH --partition=V100
#SBATCH --job-name=tr_fr_text
#SBATCH --output=freeze_text_output.txt
#SBATCH --error=freeze_text_error.txt
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00

set -x
srun python3 -u text_freezed.py
