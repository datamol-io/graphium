# Graphium Datasets

Graphium datasets are hosted at on Google Cloud Storage at `gs://graphium-public/datasets`. Graphium provides a convenient utility functions to list and download those datasets:

```python
import graphium

dataset_dir = "/my/path"
data_path = graphium.data.utils.download_graphium_dataset("graphium-zinc-micro", output_path=dataset_dir)
print(data_path)
# /my/path/graphium-zinc-micro
```

## `graphium-zinc-micro`

ADD DESCRIPTION.

- Number of molecules: xxx
- Label columns: xxx
- Split informations.

## `graphium-zinc-bench-gnn`

ADD DESCRIPTION.

- Number of molecules: xxx
- Label columns: xxx- Split informations.
