# General imports
import os
import yaml
from copy import deepcopy
import hydra
from omegaconf import DictConfig


# Current project imports
import goli
from goli.commons.config_loader import (
    config_load_constants,
    config_load_datasets, 
    config_load_gnn, 
    config_load_metrics,
    config_load_model_wrapper,
    config_load_training)


# Set up the working directory
MAIN_DIR = os.path.dirname(goli.__path__._path[0])
os.chdir(MAIN_DIR)

@hydra.main(config_name="config_micro_ZINC.yaml")
def main(cfg_main : DictConfig) -> None:
    cfg = dict(deepcopy(cfg_main))

    # Get the general parameters and generate the train/val/test datasets
    device, dtype, exp_name, cfg_gnns = config_load_constants(cfg['constants'], MAIN_DIR)
    trans, (train_dt, val_dt) = config_load_datasets(
            cfg['datasets'], main_dir=MAIN_DIR, train_val_test=['train', 'val'], device=device)

    # Initialize the network, the metrics, the model_wrapper and the trainer
    in_dim = trans.atom_dim
    out_dim = train_dt[0][1].shape[0]
    model, layer_name = config_load_gnn(cfg['model'], cfg_gnns, in_dim, out_dim, device, dtype)
    metrics, metrics_on_progress_bar = config_load_metrics(cfg['metrics'])
    model_wrapper = config_load_model_wrapper(cfg['model_wrapper'], metrics, metrics_on_progress_bar,
                                        model, layer_name, train_dt, val_dt, device, dtype)
    trainer = config_load_training(cfg['training'], model_wrapper)

    # Run the model training
    try:
        trainer.fit(model_wrapper)
        print('\n------------ TRAINING COMPLETED ------------\n\n')
    except Exception as e:
        if cfg['constants']['ignore_train_error']:
            print('\n------------ TRAINING ERROR: ------------\n\n', e)
        else:
            raise



if __name__ == "__main__":
    main()



