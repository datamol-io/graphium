"""
Unit tests for the metrics and wrappers of graphium/trainer/metrics/...
"""

import torch
import unittest as ut
import tempfile
import os
import operator as op

from graphium.trainer.metrics import (
    MetricWrapper,
    Thresholder,
)

from torchmetrics.functional import mean_squared_error


class test_Metrics(ut.TestCase):
    def test_thresholder(self):
        torch.manual_seed(42)
        preds = torch.rand(100, dtype=torch.float32)
        target = torch.rand(100, dtype=torch.float32)

        th = 0.7
        preds_greater = preds > th
        target_greater = target > th

        # Test thresholder greater
        for th_on_preds in [True, False]:
            for th_on_target in [True, False]:
                thresholder = Thresholder(
                    threshold=th, operator="greater", th_on_target=th_on_target, th_on_preds=th_on_preds
                )
                preds2, target2 = thresholder(preds, target)
                if th_on_preds:
                    self.assertListEqual(preds2.tolist(), preds_greater.tolist())
                else:
                    self.assertListEqual(preds2.tolist(), preds.tolist())
                if th_on_target:
                    self.assertListEqual(target2.tolist(), target_greater.tolist())
                else:
                    self.assertListEqual(target2.tolist(), target.tolist())

        # Test thresholder lower
        for th_on_preds in [True, False]:
            for th_on_target in [True, False]:
                thresholder = Thresholder(
                    threshold=th, operator="lower", th_on_target=th_on_target, th_on_preds=th_on_preds
                )
                preds2, target2 = thresholder(preds, target)
                if th_on_preds:
                    self.assertListEqual(preds2.tolist(), (~preds_greater).tolist())
                else:
                    self.assertListEqual(preds2.tolist(), preds.tolist())
                if th_on_target:
                    self.assertListEqual(target2.tolist(), (~target_greater).tolist())
                else:
                    self.assertListEqual(target2.tolist(), target.tolist())


class test_MetricWrapper(ut.TestCase):
    def test_target_nan_mask(self):
        torch.random.manual_seed(42)

        for sz in [(100,), (100, 1), (100, 10)]:
            err_msg = f"Error for `sz = {sz}`"

            # Generate prediction and target matrices
            target = torch.rand(sz, dtype=torch.float32)
            preds = (0.5 * target) + (0.5 * torch.rand(sz, dtype=torch.float32))
            is_nan = torch.rand(sz) > 0.3
            target = (target > 0.5).to(torch.float32)
            target[is_nan] = float("nan")

            # Compute score with different ways of ignoring NaNs
            metric = MetricWrapper(metric="mse", target_nan_mask=None)
            score1 = metric(preds, target)
            self.assertTrue(torch.isnan(score1), msg=err_msg)

            # Replace NaNs by 0
            metric = MetricWrapper(metric="mse", target_nan_mask=0)
            score2 = metric(preds, target)

            this_target = target.clone()
            this_target[is_nan] = 0
            this_preds = preds.clone()
            self.assertAlmostEqual(score2, mean_squared_error(this_preds, this_target), msg=err_msg)

            # Replace NaNs by 1.5
            metric = MetricWrapper(metric="mse", target_nan_mask=1.5)
            score3 = metric(preds, target)

            this_target = target.clone()
            this_target[is_nan] = 1.5
            this_preds = preds.clone()
            self.assertAlmostEqual(score3, mean_squared_error(this_preds, this_target), msg=err_msg)

            # Flatten matrix and ignore NaNs
            metric = MetricWrapper(metric="mse", target_nan_mask="ignore", multitask_handling="flatten")
            score4 = metric(preds, target)

            this_target = target.clone()[~is_nan]
            this_preds = preds.clone()[~is_nan]
            self.assertAlmostEqual(score4, mean_squared_error(this_preds, this_target), msg=err_msg)

            # Ignore NaNs in each column and average the score
            metric = MetricWrapper(
                metric="mse", target_nan_mask="ignore", multitask_handling="mean-per-label"
            )
            score5 = metric(preds, target)

            this_target = target.clone()
            this_preds = preds.clone()
            this_is_nan = is_nan.clone()
            if len(sz) == 1:
                this_target = target.unsqueeze(-1)
                this_preds = preds.unsqueeze(-1)
                this_is_nan = is_nan.unsqueeze(-1)

            this_target = [this_target[:, ii][~this_is_nan[:, ii]] for ii in range(this_target.shape[1])]
            this_preds = [this_preds[:, ii][~this_is_nan[:, ii]] for ii in range(this_preds.shape[1])]
            mse = []
            for ii in range(len(this_preds)):
                mse.append(mean_squared_error(this_preds[ii], this_target[ii]))
            mse = torch.mean(torch.stack(mse))
            self.assertAlmostEqual(score5.tolist(), mse.tolist(), msg=err_msg)

    def test_pickling(self):
        pickle_file = os.path.join(tempfile.gettempdir(), "test_metric_pickled.pkl")
        metrics = ["mae", "mse", mean_squared_error]
        target_nan_masks = [None, 2, "ignore"]
        multitask_handlings = [None, "flatten", "mean-per-label"]
        squeeze_targets = [True, False]
        target_to_ints = [True, False]
        other_kwargs = [{}, {"squared": False}]
        thresholds = [
            None,
            {"threshold": 0.2, "operator": "greater"},
            {"threshold": 0.3, "operator": op.lt},
            {"threshold": 0.4, "operator": "lower"},
            {"threshold": 0.5, "operator": "lower", "th_on_preds": False, "th_on_target": True},
            {"threshold": 0.6, "operator": "lower"},
        ]

        for metric in metrics:
            for target_nan_mask in target_nan_masks:
                for kwargs in other_kwargs:
                    for threshold_kwargs in thresholds:
                        for multitask_handling in multitask_handlings:
                            for squeeze_target in squeeze_targets:
                                for target_to_int in target_to_ints:
                                    err_msg = f"{metric} - {target_nan_mask} - {kwargs} - {threshold_kwargs}"

                                    if (multitask_handling is None) and (target_nan_mask == "ignore"):
                                        # Raise with incompatible options
                                        with self.assertRaises(ValueError):
                                            MetricWrapper(
                                                metric=metric,
                                                threshold_kwargs=threshold_kwargs,
                                                target_nan_mask=target_nan_mask,
                                                multitask_handling=multitask_handling,
                                                squeeze_target=squeeze_target,
                                                target_to_int=target_to_int,
                                                **kwargs,
                                            )

                                    else:
                                        metric_wrapper = MetricWrapper(
                                            metric=metric,
                                            threshold_kwargs=threshold_kwargs,
                                            target_nan_mask=target_nan_mask,
                                            multitask_handling=multitask_handling,
                                            squeeze_target=squeeze_target,
                                            target_to_int=target_to_int,
                                            **kwargs,
                                        )

                                        # Check that the metric can be saved and re-loaded without error
                                        torch.save(metric_wrapper, pickle_file)
                                        metric_wrapper2 = torch.load(pickle_file)
                                        self.assertTrue(metric_wrapper == metric_wrapper2, msg=err_msg)

                                        # Check that the metric only contains primitive types
                                        state = metric_wrapper.__getstate__()
                                        if state["threshold_kwargs"] is not None:
                                            self.assertIsInstance(
                                                state["threshold_kwargs"], dict, msg=err_msg
                                            )
                                        if isinstance(metric, str):
                                            self.assertIsInstance(state["metric"], str, msg=err_msg)

    def test_classifigression_target_squeezing(self):
        preds = torch.Tensor([[0.1, 0.1, 0.3, 0.5, 0.0, 0.1, 0.0, 0.7, 0.2, 0.0]])
        target = torch.Tensor([3, 0])
        expected_scores = [0.5, 0.75]
        n_brackets = 5
        metrics = ["accuracy", "averageprecision"]
        other_kwargs = [
            {"task": "multiclass", "num_classes": n_brackets, "top_k": 1},
            {"task": "multiclass", "num_classes": n_brackets},
        ]

        for metric, kwargs, expected_score in zip(metrics, other_kwargs, expected_scores):
            metric_wrapper = MetricWrapper(
                metric=metric,
                multitask_handling="mean-per-label",
                squeeze_targets=True,
                target_to_int=True,
                **kwargs,
            )
            score = metric_wrapper(preds, target)

            assert score == expected_score


if __name__ == "__main__":
    ut.main()
