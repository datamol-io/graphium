# `graphium-train`

To support advanced configuration, Graphium uses [`hydra`](https://hydra.cc/) to manage and write config files. A limitation of `hydra`, is that it is designed to function as the main entrypoint for a CLI application and does not easily support subcommands. For that reason, we introduced the `graphium-train` command in addition to the [`graphium`](./graphium.md) command. 

!!! info "Curious about the configs?"
    If you would like to learn more about the configs, please visit the docs [here](https://github.com/datamol-io/graphium/tree/main/expts/hydra-configs).

This page documents `graphium-train`.

## Running an experiment
To run an experiment go to the `expts/hydra-configs` folder for all available configurations. For example, to benchmark a GCN on the ToyMix dataset run
```bash
graphium-train dataset=toymix model=gcn
```
To change parameters specific to this experiment like switching from `fp16` to `fp32` precision, you can either override them directly in the CLI via
```bash
graphium-train dataset=toymix model=gcn trainer.trainer.precision=32
```
or change them permamently in the dedicated experiment config under `expts/hydra-configs/toymix_gcn.yaml`.
Integrating `hydra` also allows you to quickly switch between accelerators. E.g., running
```bash
graphium-train dataset=toymix model=gcn accelerator=gpu
```
automatically selects the correct configs to run the experiment on GPU.
Finally, you can also run a fine-tuning loop: 
```bash
graphium-train +finetuning=admet
```

To use a config file you built from scratch you can run
```bash
graphium-train --config-path [PATH] --config-name [CONFIG]
```
Thanks to the modular nature of `hydra` you can reuse many of our config settings for your own experiments with Graphium.

### Preparing the data in advance
The data preparation including the featurization (e.g., of molecules from smiles to pyg-compatible format) is embedded in the pipeline and will be performed when executing `graphium-train [...]`.

However, when working with larger datasets, it is recommended to perform data preparation in advance using a machine with sufficient allocated memory (e.g., ~400GB in the case of `LargeMix`). Preparing data in advance is also beneficial when running lots of concurrent jobs with identical molecular featurization, so that resources aren't wasted and processes don't conflict reading/writing in the same directory.

The following command-line will prepare the data and cache it, then use it to train a model.
```bash
# First prepare the data and cache it in `path_to_cached_data`
graphium data prepare ++datamodule.args.processed_graph_data_path=[path_to_cached_data]

# Then train the model on the prepared data
graphium-train [...] datamodule.args.processed_graph_data_path=[path_to_cached_data]
```

??? note "Config vs. Override"
    As with any configuration, note that `datamodule.args.processed_graph_data_path` can also be specified in the configs at `expts/hydra_configs/`.

??? note "Featurization" 
    Every time the configs of `datamodule.args.featurization` change, you will need to run a new data preparation, which will automatically be saved in a separate directory that uses a hash unique to the configs.