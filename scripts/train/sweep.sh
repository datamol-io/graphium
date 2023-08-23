#!/usr/bin/env bash

## Name of your SLURM job
#SBATCH --job-name=sweep

## Files for logs: here we redirect stoout and sterr to the same file
#SBATCH --output=outputs/sweep.out
#SBATCH --error=outputs/error_sweep.out
#SBATCH --open-mode=append

## Time limit for the job
#SBATCH --time=120:00:00

## Partition to use,
#SBATCH --partition=c112

set -e

source activate graphium_dev

wandb agent multitask-gnn/neurips2023-large-single-dataset/d2cm706t