# General imports
import os
from os.path import dirname, abspath
import yaml
import numpy as np
import pandas as pd
import torch
import fsspec

# Current project imports
import goli
from goli.config._loader import load_datamodule, load_trainer
from goli.utils.fs import mkdir
from goli.trainer.predictor import PredictorModule


# Set up the working directory
MAIN_DIR = dirname(dirname(abspath(goli.__file__)))
os.chdir(MAIN_DIR)


# MODEL_FILE = "models_checkpoints/micro_ZINC/model.ckpt"
# CONFIG_FILE = "expts/config_micro_ZINC.yaml"




def main() -> None:

    NUM_LAYERS_TO_DROP = range(4)
    DATA_NAME_ALL = ["molbace", "mollipo", "moltox21", "molHIV"]

    for data_name in DATA_NAME_ALL:
        DATA_CONFIG = f"{MAIN_DIR}/expts/config_{data_name}_pretrained.yaml"

        MODEL_PATH = "saved_models"
        MODEL_NAME = "mega_pubchem_L1000_VCAP_v4"
        MODEL_FILE = f"{MODEL_PATH}/{MODEL_NAME}/model.ckpt"
        MODEL_CONFIG = f"{MODEL_PATH}/{MODEL_NAME}/configs.yaml"

        with fsspec.open(DATA_CONFIG, "r") as f:
            data_cfg = yaml.safe_load(f)
        with fsspec.open(os.path.join(MODEL_CONFIG), "r") as f:
            model_cfg = yaml.safe_load(f)

        # Load and initialize the dataset
        data_cfg["datamodule"]["args"]["featurization"] = model_cfg["datamodule"]["args"]["featurization"]
        datamodule = load_datamodule(data_cfg)
        print("\ndatamodule:\n", datamodule, "\n")

        for num_layers_to_drop in NUM_LAYERS_TO_DROP:

            export_dir = f"predictions/fingerprints-model-{MODEL_NAME}"
            export_df_path = f"{export_dir}/{data_name}-dropped-{num_layers_to_drop}.csv.gz"

            predictor = PredictorModule.load_from_checkpoint(MODEL_FILE)
            predictor.model.drop_post_nn_layers(num_layers_to_drop=num_layers_to_drop)

            print(predictor.model)
            print(predictor.summarize(max_depth=4))

            trainer = load_trainer(data_cfg)

            # Run the model prediction
            preds = trainer.predict(model=predictor, datamodule=datamodule)
            if isinstance(preds[0], torch.Tensor):
                preds = [p.detach().cpu().numpy() for p in preds]
            preds = np.concatenate(preds, axis=0)

            # Generate output dataframe
            df = {"SMILES": datamodule.dataset.smiles}

            target = datamodule.dataset.labels
            for ii in range(target.shape[1]):
                df[f"Target-{ii}"] = target[:, ii]

            for ii in range(preds.shape[1]):
                df[f"Preds-{ii}"] = preds[:, ii]
            df = pd.DataFrame(df)
            mkdir(export_dir)
            df.to_csv(export_df_path)
            print(df)
            print(f"file saved to:`{export_df_path}`")


if __name__ == "__main__":
    main()
