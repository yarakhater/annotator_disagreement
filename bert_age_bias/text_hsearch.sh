#!/bin/bash
#SBATCH --partition=P100
#SBATCH --job-name=ab_text_hsearch
#SBATCH --output=text_hsearch_output.txt
#SBATCH --error=text_hsearch_error.txt
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00

set -x
srun python3 -u text_hsearch.py
