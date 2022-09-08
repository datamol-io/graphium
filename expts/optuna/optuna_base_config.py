def update_configuration(trial, cfg):
    # * Example of adding hyper parameter search with Optuna:
    # * https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
    cfg["architecture"]["gnn"]["hidden_dims"] = trial.suggest_int("gnn.hid", 16, 124, 16)
    cfg["architecture"]["gnn"]["depth"] = trial.suggest_int("gnn.depth", 1, 5)

    # normalization = trial.suggest_categorical("batch_norm", ["none", "batch_norm", "layer_norm"])
    # cfg["architecture"]["gnn"]["normalization"] = normalization
    # cfg["architecture"]["pre_nn"]["normalization"] = normalization
    # cfg["architecture"]["post_nn"]["normalization"] = normalization
    return cfg
