#!/bin/bash
#SBATCH --partition=V100
#SBATCH --job-name=lac_toxicity_ratings
#SBATCH --output=output.txt
#SBATCH --error=error.txt
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00

set -x
srun python3 -u lac.py