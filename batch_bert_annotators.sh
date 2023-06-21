#!/bin/bash
#SBATCH --partition=V100
#SBATCH --job-name=bert_annotator
#SBATCH --output=bert_annotator_output.txt
#SBATCH --error=bert_annotator_error.text
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --cpus-per-task=3
#SBATCH --time=10:00:00

set -x
srun python -u bert_annotators.py
