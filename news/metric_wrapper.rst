**Added:**
* `spearman_ipu` implemented to make use of `pearson_ipu`

**Changed:**

* `MetricWrapper` now accepts 2 options, `target_nan_mask` and `multitask_handling`. This makes it more versatile than the previous option
* The previous option `target_nan_mask="ignore-flatten"` is now `target_nan_mask="ignore"` with `multitask_handling="flatten"`
* The previous option `target_nan_mask="ignore-mean-label"` is now `target_nan_mask="ignore"` with `multitask_handling="mean-per-label"`
* `f1_score_ipu` and `fbeta_score_ipu` were changed to not rely on operators that do not work on IPU
