# ToyMix Baseline - Test set metrics

From the paper to be released soon. Below, you can see the baselines for the `ToyMix` dataset, a multitasking dataset comprising of `QM9`, `Zinc12k` and `Tox21`. The datasets and their splits are available on [this link](https://zenodo.org/record/7998401). The following baselines are all for models with ~150k parameters.

One can observe that the smaller datasets (`Zinc12k` and `Tox21`) beneficiate from adding another unrelated task (`QM9`), where the labels are computed from DFT simulations.

**NEW baselines added 2023/09/18**: Multitask baselines have been added for GatedGCN and MPNN++ (sum aggretator) using 3 random seeds. They achieve the best performance by a significant margin on Zinc12k and Tox21, while sacrificing a little on QM9.

| Dataset   | Model | MAE ↓     | Pearson ↑ | R² ↑     | MAE ↓   | Pearson ↑ | R² ↑   |
|-----------|-------|-----------|-----------|-----------|---------|-----------|---------|
|    | <th colspan="3" style="text-align: center;">Single-Task Model</th>  <th colspan="3" style="text-align: center;">Multi-Task Model</th>   |
|
| **QM9**   | GCN   | 0.102 ± 0.0003 | 0.958 ± 0.0007 | 0.920 ± 0.002 | 0.119 ± 0.01 | 0.955 ± 0.001 | 0.915 ± 0.001 |
|           | GIN   | 0.0976 ± 0.0006 | **0.959 ± 0.0002** | **0.922 ± 0.0004** | 0.117 ± 0.01 | 0.950 ± 0.002 | 0.908 ± 0.003 |
|           | GINE  | **0.0959 ± 0.0002** | 0.955 ± 0.002 | 0.918 ± 0.004 | 0.102 ± 0.01 | 0.956 ± 0.0009 | 0.918 ± 0.002 |
|       |   GatedGCN    |       |       |       | 0.1212 ± 0.0009 | 0.9457 ± 0.0002 | 0.8964 ± 0.0006 |
|       |   MPNN++ (sum)    |       |       |    | 0.1174 ± 0.0012 | 0.9460 ± 0.0005 | 0.8989 ± 0.0008 |
 **Zinc12k** | GCN   | 0.348 ± 0.02 | 0.941 ± 0.002 | 0.863 ± 0.01 | 0.226 ± 0.004 | 0.973 ± 0.0005 | 0.940 ± 0.003 |
|           | GIN   | 0.303 ± 0.007 | 0.950 ± 0.003 | 0.889 ± 0.003 | 0.189 ± 0.004 | 0.978 ± 0.006 | 0.953 ± 0.002 |
|           | GINE  | 0.266 ± 0.02 | 0.961 ± 0.003 | 0.915 ± 0.01 | 0.147 ± 0.009 | 0.987 ± 0.001 | 0.971 ± 0.003 |
|       | GatedGCN       |       |       |       | 0.1282 ± 0.0045 | 0.9850 ± 0.0006 | 0.9639 ± 0.0024 |
|       | MPNN++ (sum)   |       |       |       | **0.1002 ± 0.0025** | **0.9909 ± 0.0004** | **0.9777 ± 0.0014** |

|           |       | BCE ↓     | AUROC ↑ | AP ↑     | BCE ↓   | AUROC ↑ | AP ↑   |
|-----------|-------|-----------|-----------|-----------|---------|-----------|---------|
|    | <th colspan="3" style="text-align: center;">Single-Task Model</th>  <th colspan="3" style="text-align: center;">Multi-Task Model</th>   |
|
| **Tox21**   | GCN   | 0.202 ± 0.005 | 0.773 ± 0.006 | 0.334 ± 0.03 | 0.176 ± 0.001 | 0.850 ± 0.006 | 0.446 ± 0.01 |
|           | GIN   | 0.200 ± 0.002 | 0.789 ± 0.009 | 0.350 ± 0.01 | 0.176 ± 0.001 | 0.841 ± 0.005 | 0.454 ± 0.009 |
|           | GINE  | 0.201 ± 0.007 | 0.783 ± 0.007 | 0.345 ± 0.02 | 0.177 ± 0.0008 | 0.836 ± 0.004 | 0.455 ± 0.008 |
|       | GatedGCN       |       |       |       | 0.1733 ± 0.0015 | 0.8522 ± 0.0022 | **0.4620 ± 0.0118** |
|       | MPNN++ (sum)   |       |       |       | **0.1725 ± 0.0012** | **0.8569 ± 0.0005** | 0.4598 ± 0.0044 |


# LargeMix Baseline
## LargeMix test set metrics

From the paper to be released soon. Below, you can see the baselines for the `LargeMix` dataset, a multitasking dataset comprising of `PCQM4M_N4`, `PCQM4M_G25`, `PCBA_1328`, `L1000_VCAP`, and `L1000_MCF7`. The datasets and their splits are available on [this link](https://zenodo.org/record/7998401). The following baselines are all for models with 4-6M parameters.

One can observe that the smaller datasets (`L1000_VCAP` and `L1000_MCF7`) beneficiate tremendously from the multitasking. Indeed, the lack of molecular samples means that it is very easy for a model to overfit.

While `PCQM4M_G25` has no noticeable changes, the node predictions of `PCQM4M_N4` and assay predictions of `PCBA_1328` take a hit, but it is most likely due to underfitting since the training loss is also increased. It seems that 4-6M parameters is far from sufficient to capturing all of the tasks simultaneously, which motivates the need for a larger model.

| Dataset   | Model | MAE ↓     | Pearson ↑ | R² ↑     | MAE ↓   | Pearson ↑ | R² ↑   |
|-----------|-------|-----------|-----------|-----------|---------|-----------|---------|
|    | <th colspan="3" style="text-align: center;">Single-Task Model</th>  <th colspan="3" style="text-align: center;">Multi-Task Model</th>   |
|
| **Pcqm4m_g25** | GCN | 0.2362 ± 0.0003 | 0.8781 ± 0.0005 | 0.7803 ± 0.0006 | 0.2458 ± 0.0007 | 0.8701 ± 0.0002 | 0.8189 ± 0.0004 |
|               | GIN | 0.2270 ± 0.0003 | 0.8854 ± 0.0004 | 0.7912 ± 0.0006 | 0.2352 ± 0.0006 | 0.8802 ± 0.0007 | 0.7827 ± 0.0005 |
|               | GINE| **0.2223 ± 0.0007** | **0.8874 ± 0.0003** | **0.7949 ± 0.0001** | 0.2315 ± 0.0002 | 0.8823 ± 0.0002 | 0.7864 ± 0.0008 |
| **Pcqm4m_n4** | GCN | 0.2080 ± 0.0003 | 0.5497 ± 0.0010 | 0.2942 ± 0.0007 | 0.2040 ± 0.0001 | 0.4796 ± 0.0006 | 0.2185 ± 0.0002 |
|               | GIN | 0.1912 ± 0.0027 | **0.6138 ± 0.0088** | **0.3688 ± 0.0116** | 0.1966 ± 0.0003 | 0.5198 ± 0.0008 | 0.2602 ± 0.0012 |
|               | GINE| **0.1910 ± 0.0001** | 0.6127 ± 0.0003 | 0.3666 ± 0.0008 | 0.1941 ± 0.0003 | 0.5303 ± 0.0023 | 0.2701 ± 0.0034 |


|           |       | BCE ↓     | AUROC ↑ | AP ↑     | BCE ↓   | AUROC ↑ | AP ↑   |
|-----------|-------|-----------|-----------|-----------|---------|-----------|---------|
|    | <th colspan="3" style="text-align: center;">Single-Task Model</th>  <th colspan="3" style="text-align: center;">Multi-Task Model</th>   |
| <hi> | <hi> | <hi> | <hi> | <hi> | <hi> | <hi> | <hi> |
| **Pcba\_1328**    | GCN      | **0.0316 ± 0.0000** | **0.7960 ± 0.0020** | **0.3368 ± 0.0027** | 0.0349 ± 0.0002 | 0.7661 ± 0.0031 | 0.2527 ± 0.0041 |
|               | GIN      | 0.0324 ± 0.0000 | 0.7941 ± 0.0018 | 0.3328 ± 0.0019 | 0.0342 ± 0.0001 | 0.7747 ± 0.0025 | 0.2650 ± 0.0020 |
|               | GINE      | 0.0320 ± 0.0001 | 0.7944 ± 0.0023 | 0.3337 ± 0.0027 | 0.0341 ± 0.0001 | 0.7737 ± 0.0007 | 0.2611 ± 0.0043 |
| **L1000\_vcap**   | GCN      | 0.1900 ± 0.0002 | 0.5788 ± 0.0034 | 0.3708 ± 0.0007 | 0.1872 ± 0.0020 | 0.6362 ± 0.0012 | 0.4022 ± 0.0008 |
|               | GIN      | 0.1909 ± 0.0005 | 0.5734 ± 0.0029 | 0.3731 ± 0.0014 | 0.1870 ± 0.0010 | 0.6351 ± 0.0014 | 0.4062 ± 0.0001 |
|               | GINE      | 0.1907 ± 0.0006 | 0.5708 ± 0.0079 | 0.3705 ± 0.0015 | **0.1862 ± 0.0007** | **0.6398 ± 0.0043** | **0.4068 ± 0.0023** |
| **L1000\_mcf7**   | GCN      | 0.1869 ± 0.0003 | 0.6123 ± 0.0051 | 0.3866 ± 0.0010 | 0.1863 ± 0.0011 | **0.6401 ± 0.0021** | 0.4194 ± 0.0004 |
|               | GIN      | 0.1862 ± 0.0003 | 0.6202 ± 0.0091 | 0.3876 ± 0.0017 | 0.1874 ± 0.0013 | 0.6367 ± 0.0066 | **0.4198 ± 0.0036** |
|               | GINE      | **0.1856 ± 0.0005** | 0.6166 ± 0.0017 | 0.3892 ± 0.0035 | 0.1873 ± 0.0009 | 0.6347 ± 0.0048 | 0.4177 ± 0.0024 |

## LargeMix training set loss

Below is the loss on the training set. One can observe that the multi-task model always underfits the single-task, except on the two `L1000` datasets.

This is not surprising as they contain two orders of magnitude more datapoints and pose a significant challenge for the relatively small models used in this analysis. This favors the Single dataset setup (which uses a model of the same size) and we conjecture larger models to bridge this gap moving forward.

|            |       | CE or MSE loss in single-task $\downarrow$ | CE or MSE loss in multi-task $\downarrow$ |
|------------|-------|-----------------------------------------|-----------------------------------------|
|
| **Pcqm4m\_g25**    | GCN   | **0.2660 ± 0.0005** | 0.2767 ± 0.0015 |
|             | GIN   | **0.2439 ± 0.0004** | 0.2595 ± 0.0016 |
|             | GINE  | **0.2424 ± 0.0007** | 0.2568 ± 0.0012 |
|
| **Pcqm4m\_n4**    | GCN   | **0.2515 ± 0.0002** | 0.2613 ± 0.0008 |
|             | GIN   | **0.2317 ± 0.0003** | 0.2512 ± 0.0008 |
|             | GINE  | **0.2272 ± 0.0001** | 0.2483 ± 0.0004 |
|
| **Pcba\_1328**    | GCN   | **0.0284 ± 0.0010** | 0.0382 ± 0.0005 |
|             | GIN   | **0.0249 ± 0.0017** | 0.0359 ± 0.0011 |
|             | GINE  | **0.0258 ± 0.0017** | 0.0361 ± 0.0008 |
|
| **L1000\_vcap**   | GCN   | 0.1906 ± 0.0036 | **0.1854 ± 0.0148** |
|             | GIN   | 0.1854 ± 0.0030 | **0.1833 ± 0.0185** |
|             | GINE  | **0.1860 ± 0.0025** | 0.1887 ± 0.0200 |
|
| **L1000\_mcf7**   | GCN   | 0.1902 ± 0.0038 | **0.1829 ± 0.0095** |
|             | GIN   | 0.1873 ± 0.0033 | **0.1701 ± 0.0142** |
|             | GINE  | 0.1883 ± 0.0039 | **0.1771 ± 0.0010** |

## NEW: Largemix improved sweep - 2023/08-18

Unsatisfied with the prior results, we ran a bayesian search over a broader set of parameters, and including only more expressive models, namely GINE, GatedGCN and MPNN++. We further increase the number of parameters to 10M due to evidence of underfitting. We evaluate only the multitask setting.

We observe a significant improvement over all tasks, with a very notable r2-score increase of +0.53 (0.27 -> 0.80) compared to the best node-level property prediction on PCQM4M_N4.

The results are reported below over 1 seed. We are currently running more seeds of the same models.

| Dataset       | Model          | MAE ↓     | Pearson ↑ | R² ↑     |
|---------------|----------------|--------|---------|--------|
| **PCQM4M_G25**    | GINE           | 0.2250 | 0.8840  | 0.7911 |
|               | GatedGCN       | 0.2457 | 0.8698  | 0.7688 |
|               | MPNN++ (sum)   | 0.2269 | 0.8802  | 0.7855 |
|
| **PCQM4M_N4**     | GINE           | 0.2699 | 0.8475  | 0.7182 |
|               | GatedGCN       | 0.3337 | 0.8102  | 0.6566 |
|               | MPNN++ (sum)   | 0.2114 | 0.8942  | 0.8000 |

| Dataset       | Model          | BCE ↓     | AUROC ↑ | AP ↑     |
|---------------|----------------|--------|---------|--------|
| **PCBA_1328**     | GINE           | 0.0334 | 0.7879  | 0.2808 |
|               | GatedGCN       | 0.0351 | 0.7788  | 0.2611 |
|               | MPNN++ (sum)   | 0.0344 | 0.7815  | 0.2666 |
|
| **L1000_VCAP**    | GINE           | 0.1907 | 0.6416  | 0.4042 |
|               | GatedGCN       | 0.1866 | 0.6395  | 0.4092 |
|               | MPNN++ (sum)   | 0.1867 | 0.6478  | 0.4131 |
|
| **L1000_MCF7**    | GINE           | 0.1931 | 0.6352  | 0.4235 |
|               | GatedGCN       | 0.1859 | 0.6547  | 0.4224 |
|               | MPNN++ (sum)   | 0.1870 | 0.6593  | 0.4254 |



# UltraLarge Baseline

## UltraLarge test set metrics

For `UltraLarge`, we provide results for the same GNN baselines as for
`LargeMix`. Each model is trained for 50 epochs and results are averaged over 3 seeds. The remaining
setup is the same as for TOYMIX (Section E.1), reporting metrics on the Single Dataset and Multi Dataset using the same performance metrics. We further use the same models (in terms of size) as used for `LargeMix`.

For now, we report only the results for a subset representing 5% of the total dataset due to computational constraint, but aim to provide the full results soon.

Results discussion. `UltraLarge` results can be found in Table 6. Interestingly, on both graph- and node-level tasks we observe that there is no advantage of multi-tasking in terms of performance. We
expect that for this ultra-large dataset, significantly larger models are needed to successfully leverage the multi-task setup. This could be attributed to underfitting, as already demonstrated for `LargeMix`. Nonetheless, our baselines set the stage for large-scale pre-training on `UltraLarge`.

The results presented used approximately 500 GPU hours of compute, with
more compute used for development and hyperparameter search.

We further note that the graph-level tasks results are very strong. Regarding the node-level tasks, they are expected to underperform in low-parameters regime, due to clear signs of underfitting, a very large amount of labels to learn, and susceptibility to over-smoothing from traditional GNNs.


| Dataset          | Model | MAE ↓             | Pearson ↑         | R² ↑          | MAE ↓             | Pearson ↑         | R² ↑              |
|------------------|-------|-------------------|-------------------|-------------------|-------------------|-------------------|-------------------|
|    | <th colspan="3" style="text-align: center;">Single-Task Model</th>  <th colspan="3" style="text-align: center;">Multi-Task Model</th>   |
| <hi> | <hi> | <hi> | <hi> | <hi> | <hi> | <hi> | <hi> |
| **Pm6_83m_g62**      | GCN   | .2606 ± .0011     | .9004 ± .0003     | .7997 ± .0009       | .2625 ± .0011     | .8896 ± .0001     | .7982 ± .0001     |
|                  | GIN   | .2546 ± .0021     | .9051 ± .0019     | .8064 ± .0037       | .2562 ± .0000     | .8901 ± .0000     | .806 ± .0000      |
|                  | GINE  | **.2538 ± .0006** | **.9059 ± .0010** | **.8082 ± .0015**   | .258 ± .0011      | .904 ± .0000      | .8048 ± .0001     |
|
| **Pm6_83m_n7**       | GCN   | .5803 ± .0001     | .3372 ± .0004     | .1191 ± .0002      | .5971 ± .0002     | .3164 ± .0001     | .1019 ± .0011     |
|                  | GIN   | .573 ± .0002      | .3478 ± .0001     | **.1269 ± .0002**  | .5831 ± .0001     | .3315 ± .0005     | .1141 ± .0000     |
|                  | GINE  | **.572 ± .0004**  | **.3487 ± .0002** | .1266 ± .0001      | .5839 ± .0004     | .3294 ± .0002     | .1104 ± .0000     |

## UltraLarge training set loss

In the table below, we observe that the multi-task model slightly underfits the single-task model, indicating that parameters can be efficiently shared between the node-level and graph-level tasks. We further note that the training loss and the test MAE are almost equal for all tasks, indicating further benefits as we scale both the model and the data.

|                  |       | **MAE loss in single-task ↓** | **MAE loss in multi-task ↓** |
|------------------|-------|---------------------------|--------------------------|
|
| Pm6_83m_g62      | GCN   | **.2679 ± .0020**        | .2713 ± .0017           |
|                  | GIN   | **.2582 ± .0018**        | .2636 ± .0014           |
|                  | GINE  | **.2567 ± .0036**        | .2603 ± .0021           |
|
| Pm6_83m_n7       | GCN   | **.5818 ± .0021**        | .5955 ± .0023           |
|                  | GIN   | **.5707 ± .0019**        | .5851 ± .0038           |
|                  | GINE  | **.5724 ± .0015**        | .5832 ± .0027           |

