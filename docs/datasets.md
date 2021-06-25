# GOLI Datasets

GOLI datasets are hosted at on Google Cloud Storage at `gs://goli-public/datasets`. GOLI provides a convenient utility functions to list and download those datasets:

```python
import goli

dataset_dir = "/my/path"
data_path = goli.data.utils.download_goli_dataset("goli-zinc-micro", output_path=dataset_dir)
print(data_path)
# /my/path/goli-zinc-micro
```

## `goli-zinc-micro`

ADD DESCRIPTION.

- Number of molecules: xxx
- Label columns: xxx
- Split informations.

## `goli-zinc-bench-gnn`

ADD DESCRIPTION.

- Number of molecules: xxx
- Label columns: xxx- Split informations.

## `goli-htsfp`

ADD DESCRIPTION.

- Number of molecules: xxx
- Label columns: xxx
- Split informations.
