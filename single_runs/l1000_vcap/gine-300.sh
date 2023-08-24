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
#SBATCH --partition=v1001

set -e

micromamba run -n graphium -c graphium-train \
    model=gine \
    architecture=largemix \
    tasks=l1000_vcap \
    training=largemix \
    accelerator=gpu \
    trainer.model_checkpoint.dirpath="model_checkpoints/l1000_vcap/gine/300/" \
    predictor.optim_kwargs.lr=0.0004 \
    constants.seed=300 \
    constants.wandb.project="neurips2023-large-single-dataset" \
    +datamodule.args.task_specific_args.l1000_vcap.split_names=["train", "val", "test-seen"] \