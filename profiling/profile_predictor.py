from tqdm import tqdm
import os
from time import time
import yaml
import fsspec

from graphium.config._loader import (
    load_datamodule,
    load_metrics,
    load_trainer,
    load_predictor,
    load_architecture,
)
from lightning import Trainer


def main():
    CONFIG_PATH = "expts/config_micro-PCBA.yaml"
    # DATA_PATH = "https://storage.googleapis.com/graphium-public/datasets/graphium-zinc-bench-gnn/smiles_score.csv.gz"

    with fsspec.open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["datamodule"]["args"][
        "processed_graph_data_path"
    ] = "graphium/data/cache/profiling/predictor_data.cache"
    # cfg["datamodule"]["args"]["df_path"] = DATA_PATH
    cfg["trainer"]["trainer"]["max_epochs"] = 5
    cfg["trainer"]["trainer"]["min_epochs"] = 5

    datamodule = load_datamodule(cfg)

    # Initialize the network
    model_class, model_kwargs = load_architecture(
        cfg,
        in_dim_nodes=datamodule.num_node_feats_with_positional_encoding,
        in_dim_edges=datamodule.num_edge_feats,
    )

    metrics = load_metrics(cfg)
    print(metrics)
    predictor = load_predictor(cfg, model_class, model_kwargs, metrics)
    trainer = load_trainer(cfg)
    trainer.fit(model=predictor, datamodule=datamodule)

    print("Done :)")


if __name__ == "__main__":
    main()
