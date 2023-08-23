#!/usr/bin/env bash

## Name of your SLURM job
#SBATCH --job-name=test-best-th2_gine_v100

## Files for logs: here we redirect stoout and sterr to the same file
#SBATCH --output=outputs/test-best-th2_gine_v100.out
#SBATCH --error=outputs/error_test-best-th2_gine_v100.out
#SBATCH --open-mode=append

## Time limit for the job
#SBATCH --time=120:00:00

## Partition to use,
#SBATCH --partition=v1001

set -e

micromamba run -n graphium -c graphium-test architecture=largemix tasks=largemix training=largemix \
    model=gine accelerator=gpu \
    trainer.model_checkpoint.dirpath="model_checkpoints/large-dataset/th2/gine/gpu/" \
    datamodule.args.task_specific_args.l1000_vcap.df_path="expts/data/large-dataset/LINCS_L1000_VCAP_0-2_th2.csv.gz" \
    datamodule.args.task_specific_args.l1000_mcf7.df_path="expts/data/large-dataset/LINCS_L1000_MCF7_0-2_th2.csv.gz" \
    +ckpt_name_for_testing="best"
    