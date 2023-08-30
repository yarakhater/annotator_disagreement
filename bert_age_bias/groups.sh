#!/bin/bash
#SBATCH --partition=V100
#SBATCH --job-name=new_ab_groups_newhp
#SBATCH --output=new_groups_output.txt
#SBATCH --error=new_groups_error.txt
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00

set -x
srun python3 -u groups.py
