#!/bin/bash
#SBATCH --job-name=graphium_training_depth_8
#SBATCH --output=graphium_train_output_depth_8.txt
#SBATCH --error=graphium_train_error_depth_8.txt
#SBATCH --ntasks=1
#SBATCH --time=48:00:00   
#SBATCH --mem=48Gb
#SBATCH --partition=main
#SBATCH --gres=gpu:a100l:1
#SBATCH --cpus-per-task=8

# Activate the conda environment
source /home/mila/h/hussein-mohamu.jama/miniconda3/etc/profile.d/conda.sh
conda activate graphium

# Execute the training command
graphium-train model=gated_gcn accelerator=gpu +hparam_search=optuna
