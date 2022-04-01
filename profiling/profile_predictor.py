from tqdm import tqdm
import os
from time import time
import yaml
import fsspec

from goli.data.utils import load_micro_zinc
from goli.trainer.predictor import PredictorModule

from goli.config._loader import load_datamodule, load_metrics, load_trainer
from pytorch_lightning import Trainer

def main():
    MODEL_PATH = "gs://goli-private/pretrained-models/micro_model/model.ckpt"
    CONFIG_PATH = "gs://goli-private/pretrained-models/micro_model/configs.yaml"
    DATA_PATH = "https://storage.googleapis.com/goli-public/datasets/goli-zinc-bench-gnn/smiles_score.csv.gz"

    with fsspec.open(CONFIG_PATH, "r") as f:
        cfg = yaml.safe_load(f)
    cfg["datamodule"]["args"]["cache_data_path"] = "goli/data/cache/profiling_data.cache"
    cfg["datamodule"]["args"]["df_path"] = DATA_PATH
    cfg["trainer"]["trainer"]["max_epochs"] = 5
    cfg["trainer"]["trainer"]["min_epochs"] = 5

    datamodule = load_datamodule(cfg)
    predictor = PredictorModule.load_from_checkpoint(MODEL_PATH)
    trainer = load_trainer(cfg)
    trainer.fit(model=predictor, datamodule=datamodule)

    print("Done :)")


if __name__ == "__main__":
    main()