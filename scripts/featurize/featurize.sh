#!/usr/bin/env bash

## Name of your SLURM job
#SBATCH --job-name=pcba_feat

## Files for logs: here we redirect stoout and sterr to the same file
#SBATCH --output=outputs/featurize.out
#SBATCH --error=outputs/error_featurize.out
#SBATCH --open-mode=append

## Time limit for the job
#SBATCH --time=120:00:00

## Partition to use,
#SBATCH --partition=c112

set -e

micromamba run -n graphium -c graphium-prepare-data \
    architecture=largemix \
    tasks=pcba_1328 \
    training=largemix \