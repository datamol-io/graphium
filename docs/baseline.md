# ToyMix Baseline - Test set metrics

From the paper to be released soon. Below, you can see the baselines for the `ToyMix` dataset, a multitasking dataset comprising of `QM9`, `Zinc12k` and `Tox21`. The datasets and their splits are available on [this link](https://zenodo.org/record/7998401). The following baselines are all for models with ~150k parameters.

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

# LargeMix Baseline
## LargeMix test set metrics

From the paper to be released soon. Below, you can see the baselines for the `LargeMix` dataset, a multitasking dataset comprising of `PCQM4M_N4`, `PCQM4M_G25`, `PCBA_1328`, `L1000_VCAP`, and `L1000_MCF7`. The datasets and their splits are available on [this link](https://zenodo.org/record/7998401). The following baselines are all for models with 4-6M parameters.

One can observe that the smaller datasets (`L1000_VCAP` and `L1000_MCF7`) beneficiate tremendously from the multitasking. Indeed, the lack of molecular samples means that it is very easy for a model to overfit.

While `PCQM4M_G25` has no noticeable changes, the node predictions of `PCQM4M_N4` and assay predictions of `PCBA_1328` take a hit, but it is most likely due to underfitting since the training loss is also increased. It seems that 4-6M parameters is far from sufficient to capturing all of the tasks simultaneously, which motivates the need for a larger model.

| Dataset   | Model | MAE ↓     | Pearson ↑ | R² ↑     | MAE ↓   | Pearson ↑ | R² ↑   |
|-----------|-------|-----------|-----------|-----------|---------|-----------|---------|
|    | <th colspan="3" style="text-align: center;">Single-Task Model</th>  <th colspan="3" style="text-align: center;">Multi-Task Model</th>   |
| <hi> | <hi> | <hi> | <hi> | <hi> | <hi> | <hi> | <hi> |
| Pcqm4m_g25 | GCN | 0.2362 ± 0.0003 | 0.8781 ± 0.0005 | 0.7803 ± 0.0006 | 0.2458 ± 0.0007 | 0.8701 ± 0.0002 | **0.8189 ± 0.0726** |
|               | GIN | 0.2270 ± 0.0003 | 0.8854 ± 0.0004 | 0.7912 ± 0.0006 | 0.2352 ± 0.0006 | 0.8802 ± 0.0007 | 0.7827 ± 0.0005 |
|               | GINE| **0.2223 ± 0.0007** | **0.8874 ± 0.0003** | 0.7949 ± 0.0001 | 0.2315 ± 0.0002 | 0.8823 ± 0.0002 | 0.7864 ± 0.0008 |
| Pcqm4m_n4 | GCN | 0.2080 ± 0.0003 | 0.5497 ± 0.0010 | 0.2942 ± 0.0007 | 0.2040 ± 0.0001 | 0.4796 ± 0.0006 | 0.2185 ± 0.0002 |
|               | GIN | 0.1912 ± 0.0027 | **0.6138 ± 0.0088** | **0.3688 ± 0.0116** | 0.1966 ± 0.0003 | 0.5198 ± 0.0008 | 0.2602 ± 0.0012 |
|               | GINE| **0.1910 ± 0.0001** | 0.6127 ± 0.0003 | 0.3666 ± 0.0008 | 0.1941 ± 0.0003 | 0.5303 ± 0.0023 | 0.2701 ± 0.0034 |


|           |       | BCE ↓     | AUROC ↑ | AP ↑     | BCE ↓   | AUROC ↑ | AP ↑   |
|-----------|-------|-----------|-----------|-----------|---------|-----------|---------|
|    | <th colspan="3" style="text-align: center;">Single-Task Model</th>  <th colspan="3" style="text-align: center;">Multi-Task Model</th>   |
| <hi> | <hi> | <hi> | <hi> | <hi> | <hi> | <hi> | <hi> |
| Pcba\_1328    | GCN      | **0.0316 ± 0.0000** | **0.7960 ± 0.0020** | **0.3368 ± 0.0027** | 0.0349 ± 0.0002 | 0.7661 ± 0.0031 | 0.2527 ± 0.0041 |
|               | GIN      | 0.0324 ± 0.0000 | 0.7941 ± 0.0018 | 0.3328 ± 0.0019 | 0.0342 ± 0.0001 | 0.7747 ± 0.0025 | 0.2650 ± 0.0020 |
|               | GINE      | 0.0320 ± 0.0001 | 0.7944 ± 0.0023 | 0.3337 ± 0.0027 | 0.0341 ± 0.0001 | 0.7737 ± 0.0007 | 0.2611 ± 0.0043 |
| L1000\_vcap   | GCN      | 0.1900 ± 0.0002 | 0.5788 ± 0.0034 | 0.3708 ± 0.0007 | 0.1872 ± 0.0020 | 0.6362 ± 0.0012 | 0.4022 ± 0.0008 |
|               | GIN      | 0.1909 ± 0.0005 | 0.5734 ± 0.0029 | 0.3731 ± 0.0014 | 0.1870 ± 0.0010 | 0.6351 ± 0.0014 | 0.4062 ± 0.0001 |
|               | GINE      | 0.1907 ± 0.0006 | 0.5708 ± 0.0079 | 0.3705 ± 0.0015 | **0.1862 ± 0.0007** | **0.6398 ± 0.0043** | **0.4068 ± 0.0023** |
| L1000\_mcf7   | GCN      | 0.1869 ± 0.0003 | 0.6123 ± 0.0051 | 0.3866 ± 0.0010 | 0.1863 ± 0.0011 | **0.6401 ± 0.0021** | 0.4194 ± 0.0004 |
|               | GIN      | 0.1862 ± 0.0003 | 0.6202 ± 0.0091 | 0.3876 ± 0.0017 | 0.1874 ± 0.0013 | 0.6367 ± 0.0066 | **0.4198 ± 0.0036** |
|               | GINE      | **0.1856 ± 0.0005** | 0.6166 ± 0.0017 | 0.3892 ± 0.0035 | 0.1873 ± 0.0009 | 0.6347 ± 0.0048 | 0.4177 ± 0.0024 |

## LargeMix training set loss

Below is the loss on the training set. One can observe that the multi-task model always underfits the single-task, except on the two `L1000` datasets.

This is not surprising as they contain two orders of magnitude more datapoints and pose a significant challenge for the relatively small models used in this analysis. This favors the Single dataset setup (which uses a model of the same size) and we conjecture larger models to bridge this gap moving forward.

|            |       | CE or BCE loss in single-task $\downarrow$ | CE or BCE loss in multi-task $\downarrow$ |
|------------|-------|-----------------------------------------|-----------------------------------------|
|            |       |                                       |                                       |
| **Pcqm4m\_g25**    | GCN   | **0.2660 ± 0.0005** | 0.2767 ± 0.0015 |
|             | GIN   | **0.2439 ± 0.0004** | 0.2595 ± 0.0016 |
|             | GINE  | **0.2424 ± 0.0007** | 0.2568 ± 0.0012 |
|            |       |                                       |                                       |
| **Pcqm4m\_n4**    | GCN   | **0.2515 ± 0.0002** | 0.2613 ± 0.0008 |
|             | GIN   | **0.2317 ± 0.0003** | 0.2512 ± 0.0008 |
|             | GINE  | **0.2272 ± 0.0001** | 0.2483 ± 0.0004 |
|            |       |                                       |                                       |
| **Pcba\_1328**    | GCN   | **0.0284 ± 0.0010** | 0.0382 ± 0.0005 |
|             | GIN   | **0.0249 ± 0.0017** | 0.0359 ± 0.0011 |
|             | GINE  | **0.0258 ± 0.0017** | 0.0361 ± 0.0008 |
|            |       |                                       |                                       |
| **L1000\_vcap**   | GCN   | 0.1906 ± 0.0036 | **0.1854 ± 0.0148** |
|             | GIN   | 0.1854 ± 0.0030 | **0.1833 ± 0.0185** |
|             | GINE  | **0.1860 ± 0.0025** | 0.1887 ± 0.0200 |
|            |       |                                       |                                       |
| **L1000\_mcf7**   | GCN   | 0.1902 ± 0.0038 | **0.1829 ± 0.0095** |
|             | GIN   | 0.1873 ± 0.0033 | **0.1701 ± 0.0142** |
|             | GINE  | 0.1883 ± 0.0039 | **0.1771 ± 0.0010** |

# UltraLarge Baseline
Coming soon!

