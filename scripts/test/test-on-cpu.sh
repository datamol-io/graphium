#!/usr/bin/env bash

## Name of your SLURM job
#SBATCH --job-name=test-on-cpu

## Files for logs: here we redirect stoout and sterr to the same file
#SBATCH --output=outputs/name=test-on-cpu.out
#SBATCH --error=outputs/error_name=test-on-cpu.out
#SBATCH --open-mode=append

## Time limit for the job
#SBATCH --time=120:00:00

## Partition to use,
#SBATCH --partition=c112

set -e

micromamba run -n graphium -c graphium-test architecture=largemix tasks=largemix training=largemix \
    accelerator=cpu \
    trainer.model_checkpoint.dirpath="model_checkpoints/large-dataset/th4/gcn/gpu/" \
    datamodule.args.task_specific_args.l1000_vcap.df_path="expts/data/large-dataset/LINCS_L1000_VCAP_0-2_th4.csv.gz" \
    datamodule.args.task_specific_args.l1000_mcf7.df_path="expts/data/large-dataset/LINCS_L1000_MCF7_0-2_th4.csv.gz" \
    model=gcn +ckpt_name_for_testing="best"
    