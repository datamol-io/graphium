#!/bin/bash
#SBATCH --job-name=graphium_training_main
#SBATCH --output=graphium_train_output_main.txt
#SBATCH --error=graphium_train_error_main.txt
#SBATCH --ntasks=1
#SBATCH --time=96:00:00   
#SBATCH --mem=48Gb
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8

# Activate the conda environment
source /home/mila/h/hussein-mohamu.jama/miniconda3/etc/profile.d/conda.sh
conda activate graphium

# Execute the training command
wandb agent --count 10 jmohamud/toymix_baselines/23je914w
