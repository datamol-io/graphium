import yaml

import goli


def test_load_pretrained_model():
    predictor = goli.trainer.PredictorModule.load_pretrained_models("goli-zinc-micro-dummy-test")
    assert isinstance(predictor, goli.trainer.predictor.PredictorModule)


def test_training(datadir, tmpdir):

    config_path = datadir / "config_micro_ZINC.yaml"
    data_path = datadir / "micro_ZINC.csv"

    # Load a config
    with open(config_path, "r") as file:
        yaml_config = yaml.load(file, Loader=yaml.FullLoader)

    training_dir = tmpdir.mkdir("training")

    # Tweak config and paths
    yaml_config["datamodule"]["args"]["df_path"] = data_path
    yaml_config["datamodule"]["args"]["cache_data_path"] = None

    yaml_config["trainer"]["trainer"]["min_epochs"] = 1
    yaml_config["trainer"]["trainer"]["max_epochs"] = 1

    yaml_config["trainer"]["logger"]["save_dir"] = training_dir
    yaml_config["trainer"]["model_checkpoint"]["dirpath"] = None
    yaml_config["trainer"]["trainer"]["default_root_dir"] = training_dir
    yaml_config["trainer"]["trainer"]["gpus"] = 0
    yaml_config["trainer"]["trainer"]["limit_train_batches"] = 1
    yaml_config["trainer"]["trainer"]["limit_val_batches"] = 1
    yaml_config["trainer"]["trainer"]["limit_test_batches"] = 1

    # Load datamodule
    datamodule = goli.config.load_datamodule(yaml_config)

    # Load a trainer
    trainer = goli.config.load_trainer(yaml_config)

    # Load a pretrained model
    predictor = goli.trainer.PredictorModule.load_pretrained_models("goli-zinc-micro-dummy-test")

    # Inference
    results = trainer.predict(predictor, datamodule=datamodule, return_predictions=True)

    assert len(results) == 8  # type: ignore
    assert tuple(results[0].shape) == (128, 1)  # type: ignore
