# `graphium`

**Usage**:

```console
$ graphium [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `data`: Graphium datasets.
* `finetune`: Utility CLI for extra fine-tuning utilities.

## `graphium data`

Graphium datasets.

**Usage**:

```console
$ graphium data [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `download`: Download a Graphium dataset.
* `list`: List available Graphium dataset.
* `prepare`: Prepare a Graphium dataset.

### `graphium data download`

Download a Graphium dataset.

**Usage**:

```console
$ graphium data download [OPTIONS] NAME OUTPUT
```

**Arguments**:

* `NAME`: [required]
* `OUTPUT`: [required]

**Options**:

* `--progress / --no-progress`: [default: progress]
* `--help`: Show this message and exit.

### `graphium data list`

List available Graphium dataset.

**Usage**:

```console
$ graphium data list [OPTIONS]
```

**Options**:

* `--help`: Show this message and exit.

### `graphium data prepare`

Prepare a Graphium dataset.

**Usage**:

```console
$ graphium data prepare [OPTIONS] OVERRIDES...
```

**Arguments**:

* `OVERRIDES...`: [required]

**Options**:

* `--help`: Show this message and exit.

## `graphium finetune`

Utility CLI for extra fine-tuning utilities.

**Usage**:

```console
$ graphium finetune [OPTIONS] COMMAND [ARGS]...
```

**Options**:

* `--help`: Show this message and exit.

**Commands**:

* `admet`: Utility CLI to easily fine-tune a model on...
* `fingerprint`: Endpoint for getting fingerprints from a...

### `graphium finetune admet`

Utility CLI to easily fine-tune a model on (a subset of) the benchmarks in the TDC ADMET group.

A major limitation is that we cannot use all features of the Hydra CLI, such as multiruns.

**Usage**:

```console
$ graphium finetune admet [OPTIONS] OVERRIDES...
```

**Arguments**:

* `OVERRIDES...`: [required]

**Options**:

* `--name TEXT`
* `--inclusive-filter / --no-inclusive-filter`: [default: inclusive-filter]
* `--help`: Show this message and exit.

### `graphium finetune fingerprint`

Endpoint for getting fingerprints from a pretrained model.

The pretrained model should be a `.ckpt` path or pre-specified, named model within Graphium.
The fingerprint layer specification should be of the format `module:layer`.
If specified as a list, the fingerprints from all the specified layers will be concatenated.
See the docs of the `graphium.finetuning.fingerprinting.Fingerprinter` class for more info.

**Usage**:

```console
$ graphium finetune fingerprint [OPTIONS] FINGERPRINT_LAYER_SPEC... PRETRAINED_MODEL SAVE_DESTINATION
```

**Arguments**:

* `FINGERPRINT_LAYER_SPEC...`: [required]
* `PRETRAINED_MODEL`: [required]
* `SAVE_DESTINATION`: [required]

**Options**:

* `--output-type TEXT`: Either numpy (.npy) or torch (.pt) output  [default: torch]
* `-o, --override TEXT`: Hydra overrides
* `--help`: Show this message and exit.
