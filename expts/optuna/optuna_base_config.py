def update_configuration(trial, cfg):
    # * Example of adding hyper parameter search with Optuna:
    # * https://optuna.readthedocs.io/en/stable/reference/generated/optuna.trial.Trial.html
    cfg["architecture"]["gnn"]["hidden_dims"] = trial.suggest_int("gnn.hid", 16, 124, step=16)
    cfg["architecture"]["gnn"]["depth"] = trial.suggest_int("gnn.depth", 1, 5)

    # normalization = trial.suggest_categorical("batch_norm", ["none", "batch_norm", "layer_norm"])
    # cfg["architecture"]["gnn"]["normalization"] = normalization
    # cfg["architecture"]["pre_nn"]["normalization"] = normalization
    # cfg["architecture"]["graph_output_nn"]["normalization"] = normalization
    # feature_count = embed_dim * 7 #7
    # hidden_neurons = trial.suggest_categorical("hidden_neurons", [[16, 32, 16], [16, 32, 16], [16, 32, 64, 32, 16], [16, 32, 64, 128, 64, 32, 16]]) #[64, 32], [16, 32, 16]
    # bidirectional = trial.suggest_categorical("bidirectional", [True, False]) # True
    # attention_style = trial.suggest_categorical("attention_style", ['Luong', 'Bahdanau']) # Bahdanau, Luong

    return cfg
