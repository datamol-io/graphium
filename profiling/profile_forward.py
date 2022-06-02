from tqdm import tqdm
import os
from time import time
import torch
import numpy as np
import yaml
import fsspec
import dgl

from goli.data.utils import load_micro_zinc
from goli.trainer.predictor import PredictorModule

from goli.config._loader import load_datamodule, load_metrics, load_trainer
from pytorch_lightning import Trainer


def main():
    MODEL_PATH = "gs://goli-private/pretrained-models/micro_model/model.ckpt"
    CONFIG_PATH = "gs://goli-private/pretrained-models/micro_model/configs.yaml"
    DATA_PATH = "https://storage.googleapis.com/goli-public/datasets/goli-zinc-bench-gnn/smiles_score.csv.gz"
    BATCH_SIZE = 1000

    with fsspec.open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["datamodule"]["args"]["cache_data_path"] = "goli/data/cache/profiling_forward_data.cache"
    cfg["datamodule"]["args"]["df_path"] = DATA_PATH
    cfg["datamodule"]["args"]["sample_size"] = BATCH_SIZE
    cfg["datamodule"]["args"]["prepare_dict_or_graph"] = "dglgraph"

    datamodule = load_datamodule(cfg)
    datamodule.prepare_data()
    datamodule.setup()

    dglgraphs = [datamodule.train_ds[ii]["features"] for ii in range(len(datamodule.train_ds))]
    dglgraphs = dgl.batch(dglgraphs)
    dglgraphs.ndata["feat"] = dglgraphs.ndata["feat"].to(dtype=torch.float32)
    dglgraphs.edata["feat"] = dglgraphs.edata["feat"].to(dtype=torch.float32)
    with fsspec.open(MODEL_PATH) as f:
        model = torch.load(f, map_location="cpu")
    predictor = PredictorModule.load_from_checkpoint(MODEL_PATH)
    model = predictor.model

    start = time()
    for ii in range(10):
        print(ii)
        model.forward(dglgraphs)
    print(time() - start)

    print("Done :)")


if __name__ == "__main__":
    main()
