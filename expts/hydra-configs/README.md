# Configuring Graphium with Hydra
This document provides users with a point of entry to composing configs in Graphium. As a flexible library with many features, configuration is an important part of Graphium. To make configurations as reusable as possible while providing maximum flexibility, we integrated Graphium with `hydra`. Our config structure is designed to make the following functionality as accessible as possible:

- Switching between **accelerators** (CPU, GPU and IPU)
- **Benchmarking** different models on the same dataset
- **Fine-tuning** a pre-trained model on a new dataset

In what follows, we describe how each of the above functionality is achieved and how users can benefit from this design to achieve the most with Graphium with as little configuration as possible.

## Accelerators
With Graphium supporting CPU, GPU and IPU hardware, easily switching between these accelerators is pre-configured. General, accelerator-specific configs are specified under `accelerator/`, whereas experiment-specific differences between the accelerators are specialized under `training/accelerator`.

## Benchmarking
Benchmarking multiple models on the same datasets and tasks requires us to easily switch between model configurations without redefining major parts of the architecture, task heads, featurization, metrics, predictor, etc. For example, when changing from a GCN to a GIN model, a simple switch of `architecture.gnn.layer_type: 'pyg:gin'` might suffice. Hence, we abstract the `model` configs under `model/` where such model configurations can be specified.
In addition, switching models may have implications on configs specific to your current experiment, such as the name of the run or the directory to which model checkpoints are written. To enable such overrides, we can utilize `hydra` [specializations](https://hydra.cc/docs/patterns/specializing_config/). For example, for our ToyMix dataset, we specify the layer type under `model/[model_name].yaml`, e.g., for the GCN layer,

```yaml
# @package _global_

architecture:
  gnn:
    layer_type: 'pyg:gcn'
```

and set experiment-related parameters in `training/model/toymix_[model_name].yaml` as a specialization, e.g., for the GIN layer,

```yaml
# @package _global_

constants:
  name: neurips2023_small_data_gin
  ...

trainer:
  model_checkpoint:
    dirpath: models_checkpoints/neurips2023-small-gin/${now:%Y-%m-%d_%H-%M-%S}/
```
We can now utilize `hydra` to e.g., run a sweep over our models on the ToyMix dataset via

```bash
graphium-train -m model=gcn,gin
```
where the ToyMix dataset is pre-configured in `main.yaml`. Read on to find out how to define new datasets and architectures for pre-training and fine-tuning.

## Pre-training / Fine-tuning
Say you trained a model with the following command:
```bash
graphium-train --config-name "main"
```

Fine-tuning this model on downstream tasks is then as simple as:
```bash
graphium-train --config-name "main" +finetuning=...
```

From a configuration point-of-view, fine-tuning requires us to load a pre-trained model and override part of the training configuration to fine-tune it on downstream tasks. To allow a quick switch between pre-training and fine-tuning, by default, we configure models and the corresponding tasks in a separate manner. More specifically,

- under `architecture/` we store architecture related configurations such as the definition of the GNN/Transformer layers or positional/structural encoders
- under `tasks/` we store configurations specific to one task set, such as the multi-task dataset ToyMix
  - under `tasks/task_heads` we specify the task-specific heads to add on top of the base architecture.
  - under `tasks/loss_metrics_datamodule` we specify the data-module to use and the task-specific loss functions and metrics
- under `training/` we store configurations specific to training models which could be different for each combination of `architecture` and `tasks`
- under `finetuning/` we store configurations with overrides

Since architecture and tasks are logically separated it now becomes very easy to e.g., use an existing architecture backbone on a new set of tasks or a new dataset altogether. Additionally, separating training allows us to specify different training parameters for e.g., pre-training and fine-tuning of the same architecture and task set.

We will now detail how you can add new architectures, tasks and training configurations.

### Adding an architecture
The architecture config consists of specifications of the neural network components, including encoders, under the config key `architecture` and the featurization, containing the positional/structural information that is to be extracted from the data.
To add a new architecture, create a file `architecture/my_architecture.yaml` with the following information specified:
```yaml
# @package _global_
architecture:
  model_type: FullGraphMultiTaskNetwork # for example
  pre_nn:
    ...

  pre_nn_edges:
    ...

  pe_encoders:
    encoders: # your encoders
      ...

  gnn: # your GNN definition
    ...

  graph_output_nn: # output NNs for different levels such as graph, node, etc.
    graph:
      ...
    node:
      ...
    ...

datamodule:
  module_type: "MultitaskFromSmilesDataModule"
  args: # Make sure to not specify anything task-specific here
    ...
  featurization:
    ...
```
You can then select your new architecture during training, e.g., by running
```bash
graphium-train architecture=my_architecture
```

### Adding tasks
The task set config consists of specifications for the task head neural nets under the config key `architecture.task_heads`; if required, any task-specific arguments to the datamodule you use, e.g., `datamodule.args.task_specfic_args` when using the `MultitaskFromSmilesDataModule` datamodule; the per-task metrics under the config key `metrics.[task]` where `[task]` matches the tasks specified under `architecture.task_heads`; the per-task configs of the `predictor` module, as well as the loss functions of the task set under the config key `predictor.loss_fun`.
To add a new task set, create a file `tasks/my_tasks.yaml` with the following information specified:
```yaml
# @package _global_
architecture:
    task_heads:
        task1:
            ...
        task2:
            ...

datamodule: # optional, depends on your concrete datamodule class. Here: "MultitaskFromSmilesDataModule"
    args:
        task_specific_args:
            task1:
                ...
            task2:
                ...

metrics:
    task1:
        ...
    task2:
        ...

predictor:
  metrics_on_progress_bar:
    task1:
    task2:
  loss_fun: ... # your loss functions for the multi-tasking
```
You can then select your new dataset during training, e.g., by running
```bash
graphium-train tasks=my_tasks
```

### Adding training configs
The training configs consist of specifications to the `predictor` and `trainer` modules.
To add new training configs, create a file `training/my_training.yaml` with the following information specified:
```yaml
# @package _global_
predictor:
    optim_kwargs:
    lr: 4.e-5
    torch_scheduler_kwargs: # example
        module_type: WarmUpLinearLR
        max_num_epochs: &max_epochs 100
        warmup_epochs: 10
        verbose: False
    scheduler_kwargs:
        ...

trainer:
  ...
  trainer: # example
    precision: 16
    max_epochs: *max_epochs
    min_epochs: 1
    check_val_every_n_epoch: 20
```
