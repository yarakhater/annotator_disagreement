#!/bin/bash
#SBATCH --partition=P100
#SBATCH --job-name=fr_new_ab_annotators
#SBATCH --output=fr_new_annotators_output.txt
#SBATCH --error=fr_new_annotators_error.txt
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00

set -x
srun python3 -u new_annotators_freezed.py
