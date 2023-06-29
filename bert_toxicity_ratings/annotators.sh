#!/bin/bash
#SBATCH --partition=V100
#SBATCH --job-name=tr_annotators
#SBATCH --output=freeze_bert_annotators_output.txt
#SBATCH --error=freeze_bert_annotators_error.txt
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --cpus-per-task=3
#SBATCH --time=48:00:00

set -x
srun python3 -u annotators.py
