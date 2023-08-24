#!/usr/bin/env bash

## Name of your SLURM job
#SBATCH --job-name=train

## Files for logs: here we redirect stoout and sterr to the same file
#SBATCH --output=outputs/train.out
#SBATCH --error=outputs/error_train.out
#SBATCH --open-mode=append

## Time limit for the job
#SBATCH --time=120:00:00

## Partition to use,
#SBATCH --partition=v1004

set -e

micromamba run -n graphium -c graphium-train \
    model=gin \
    architecture=largemix \
    tasks=pcba_1328 \
    training=largemix \
    accelerator=gpu \
    trainer.model_checkpoint.dirpath="model_checkpoints/pcba_1328/gin/300/" \
    predictor.optim_kwargs.lr=0.0004 \
    constants.seed=600 \
    constants.wandb.project="neurips2023-large-single-dataset" \