# General imports
import os
from os.path import dirname, abspath
import yaml
from copy import deepcopy
from omegaconf import DictConfig
import timeit
from loguru import logger
from datetime import datetime
from pytorch_lightning.utilities.model_summary import ModelSummary

# Current project imports
import goli
from goli.config._loader import (
    load_datamodule,
    load_architecture,
    load_predictor,
    load_trainer,
)
from goli.utils.safe_run import SafeRun




# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(goli.__file__)))
CONFIG_FILE = "expts/configs/config_ipu_qm9_bug.yaml"
os.chdir(MAIN_DIR)

import torch
from goli.ipu.to_dense_batch import to_dense_batch, to_sparse_batch
from torch_scatter import scatter
from goli.utils.mup import set_base_shapes

class MiniModuleWithScatterBug(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(55, 32)
        self.attn_layer = torch.nn.MultiheadAttention(embed_dim=32, num_heads=2)


    def _sa_block(self, x, attn_mask, key_padding_mask):
        """Self-attention block."""
        x = self.attn_layer(
            x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False
        )[0]
        return x

    def forward(self, batch):
        h = self.linear(batch["feat"])

        # Convert the tensor to a dense batch, then back to a sparse batch
        h_dense, mask, idx = to_dense_batch(
            h, batch.batch, max_num_nodes_per_graph=10, drop_nodes_last_graph=True
        )
        h_attn = self._sa_block(h_dense, None, ~mask.T) # Dont know why I need to transpose the mask?
        h_attn = to_sparse_batch(h_attn, idx)
        h_out = scatter(h_attn, batch.batch, dim=0, dim_size=None, reduce="mean")
        h_out = h_out[:, :2]
        return {"homo": h_out}


def main(cfg: DictConfig, run_name: str = "main", add_date_time: bool = True) -> None:
    st = timeit.default_timer()

    if add_date_time:
        run_name += "_" + datetime.now().strftime("%d.%m.%Y_%H.%M.%S")

    cfg = deepcopy(cfg)

    # Load and initialize the dataset
    datamodule = load_datamodule(cfg)

    # Initialize the network
    model_class, model_kwargs = load_architecture(
        cfg,
        in_dims=datamodule.in_dims,
    )

    metrics = {"homo": {}}
    predictor = load_predictor(cfg, model_class, model_kwargs, metrics)
    model = MiniModuleWithScatterBug()
    predictor.model = set_base_shapes(model, None)

    logger.info(predictor.model)
    logger.info(ModelSummary(predictor, max_depth=4))

    trainer = load_trainer(cfg, run_name)

    datamodule.prepare_data()
    # Run the model training
    with SafeRun(name="TRAINING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
        trainer.fit(model=predictor, datamodule=datamodule)

    with SafeRun(name="TESTING", raise_error=cfg["constants"]["raise_train_error"], verbose=True):
        trainer.test(model=predictor, datamodule=datamodule)  # , ckpt_path=ckpt_path)

    logger.info("--------------------------------------------")
    logger.info("total computation used", timeit.default_timer() - st)
    logger.info("--------------------------------------------")

    return trainer.callback_metrics


if __name__ == "__main__":
    with open(os.path.join(MAIN_DIR, CONFIG_FILE), "r") as f:
        cfg = yaml.safe_load(f)
    main(cfg)
