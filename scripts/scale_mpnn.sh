#!/bin/bash

graphium-train \
    --config-path=/home/frederik_valencediscovery_com/projects/graphium_hps/expts/configs/ \
    --config-name=config_mpnn_base.yaml \
    constants.max_epochs=100 \
    trainer.model_checkpoint.dirpath=model_checkpoints/large-dataset/scale_mpnn/ \
    +architecture.mup_scale_factor=2 +architecture.mup_base_path=mup/mpnn_base/base_shapes.yaml \
    datamodule.args.batch_size_inference=1024 datamodule.args.batch_size_training=1024 +trainer.trainer.accumulate_grad_batches=2 \