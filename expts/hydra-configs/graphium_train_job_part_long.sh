#!/bin/bash
#SBATCH --job-name=graphium_training_long
#SBATCH --output=graphium_train_output_long.txt
#SBATCH --error=graphium_train_error_long.txt
#SBATCH --ntasks=1
#SBATCH --time=48:00:00   
#SBATCH --mem=48Gb
#SBATCH --partition=long
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8

# Activate the conda environment
source /home/mila/h/hussein-mohamu.jama/miniconda3/etc/profile.d/conda.sh
conda activate graphium

# Execute the training command
wandb agent --count 10 jmohamud/toymix_baselines/px0i2anh
