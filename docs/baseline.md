# ToyMix Baseline

From the paper to be released soon. Below, you can see the baselines for the `ToyMix` dataset, a multitasking dataset comprising of `QM9`, `Zinc12k` and `Tox21`. The datasets and their splits are available on [this link](https://zenodo.org/record/7998401).

One can observe that the smaller datasets (`Zinc12k` and `Tox21`) beneficiate from adding another unrelated task (`QM9`), where the labels are computed from DFT simulations.

| Dataset   | Model | MAE ↓     | Pearson ↑ | R² ↑     | MAE ↓   | Pearson ↑ | R² ↑   |
|-----------|-------|-----------|-----------|-----------|---------|-----------|---------|
|    | <th colspan="3" style="text-align: center;">Single-Task Model</th>  <th colspan="3" style="text-align: center;">Multi-Task Model</th>   |
| <hi> | <hi> | <hi> | <hi> | <hi> | <hi> | <hi> | <hi> |
| **QM9**   | GCN   | 0.102 ± 0.0003 | 0.958 ± 0.0007 | 0.920 ± 0.002 | 0.119 ± 0.01 | 0.955 ± 0.001 | 0.915 ± 0.001 |
|           | GIN   | 0.0976 ± 0.0006 | **0.959 ± 0.0002** | **0.922 ± 0.0004** | 0.117 ± 0.01 | 0.950 ± 0.002 | 0.908 ± 0.003 |
|           | GINE  | **0.0959 ± 0.0002** | 0.955 ± 0.002 | 0.918 ± 0.004 | 0.102 ± 0.01 | 0.956 ± 0.0009 | 0.918 ± 0.002 |
| <hi> | <hi> | <hi> | <hi> | <hi> | <hi> | <hi> | <hi> |
| **Zinc12k** | GCN   | 0.348 ± 0.02 | 0.941 ± 0.002 | 0.863 ± 0.01 | 0.226 ± 0.004 | 0.973 ± 0.0005 | 0.940 ± 0.003 |
|           | GIN   | 0.303 ± 0.007 | 0.950 ± 0.003 | 0.889 ± 0.003 | 0.189 ± 0.004 | 0.978 ± 0.006 | 0.953 ± 0.002 |
|           | GINE  | 0.266 ± 0.02 | 0.961 ± 0.003 | 0.915 ± 0.01 | **0.147 ± 0.009** | **0.987 ± 0.001** | **0.971 ± 0.003** |

|           |       | BCE ↓     | AUROC ↑ | AP ↑     | BCE ↓   | AUROC ↑ | AP ↑   |
|-----------|-------|-----------|-----------|-----------|---------|-----------|---------|
|    | <th colspan="3" style="text-align: center;">Single-Task Model</th>  <th colspan="3" style="text-align: center;">Multi-Task Model</th>   |
| <hi> | <hi> | <hi> | <hi> | <hi> | <hi> | <hi> | <hi> |
| **Tox21**   | GCN   | 0.202 ± 0.005 | 0.773 ± 0.006 | 0.334 ± 0.03 | **0.176 ± 0.001** | **0.850 ± 0.006** | 0.446 ± 0.01 |
|           | GIN   | 0.200 ± 0.002 | 0.789 ± 0.009 | 0.350 ± 0.01 | 0.176 ± 0.001 | 0.841 ± 0.005 | 0.454 ± 0.009 |
|           | GINE  | 0.201 ± 0.007 | 0.783 ± 0.007 | 0.345 ± 0.02 | 0.177 ± 0.0008 | 0.836 ± 0.004 | **0.455 ± 0.008** |
