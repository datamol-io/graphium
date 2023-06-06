"""
Unit tests for the metrics and wrappers of graphium/utils/...
"""

from graphium.utils.tensor import nan_mad, nan_mean, nan_std, nan_var, nan_median
from graphium.utils.safe_run import SafeRun
import torch
import numpy as np
import scipy as sp
import unittest as ut


class test_nan_statistics(ut.TestCase):
    torch.manual_seed(42)

    dims = [
        None,
        (0),
        (1),
        (2),
        (-1),
        (-2),
        (-3),
        (0, 1),
        (0, 2),
    ]
    # Create tensor
    sz = (10, 6, 7)
    tensor = torch.randn(sz, dtype=torch.float32) ** 2 + 3
    is_nan = torch.rand(sz) > 0.4
    tensor[is_nan] = float("nan")

    def test_nan_mean(self):
        for keepdim in [False, True]:
            for dim in self.dims:
                err_msg = f"Error for :\n dim = {dim}\n keepdim = {keepdim}"

                tensor = self.tensor.clone()

                # Prepare the arguments for numpy vs torch
                if dim is not None:
                    torch_kwargs = {"dim": dim, "keepdim": keepdim}
                    numpy_kwargs = {"axis": dim, "keepdims": keepdim}
                else:
                    torch_kwargs = {}
                    numpy_kwargs = {}

                # Compare the nan-mean
                torch_mean = nan_mean(tensor, **torch_kwargs)
                numpy_mean = np.nanmean(tensor.numpy(), **numpy_kwargs)
                np.testing.assert_almost_equal(torch_mean.numpy(), numpy_mean, decimal=6, err_msg=err_msg)

    def test_nan_std_var(self):
        for unbiased in [True, False]:
            for keepdim in [False, True]:
                for dim in self.dims:
                    err_msg = f"Error for :\n\tdim = {dim}\n\tkeepdim = {keepdim}\n\tunbiased = {unbiased}"

                    tensor = self.tensor.clone()

                    # Prepare the arguments for numpy vs torch
                    if dim is not None:
                        torch_kwargs = {"dim": dim, "keepdim": keepdim, "unbiased": unbiased}
                        numpy_kwargs = {"axis": dim, "keepdims": keepdim, "ddof": float(unbiased)}
                    else:
                        torch_kwargs = {"unbiased": unbiased}
                        numpy_kwargs = {"ddof": float(unbiased)}

                    # Compare the std
                    torch_std = nan_std(tensor, **torch_kwargs)
                    numpy_std = np.nanstd(tensor.numpy(), **numpy_kwargs)
                    np.testing.assert_almost_equal(torch_std.numpy(), numpy_std, decimal=6, err_msg=err_msg)

                    # Compare the variance
                    torch_var = nan_var(tensor, **torch_kwargs)
                    numpy_var = np.nanvar(tensor.numpy(), **numpy_kwargs)
                    np.testing.assert_almost_equal(torch_var.numpy(), numpy_var, decimal=6, err_msg=err_msg)

    def test_nan_median(self):
        for keepdim in [False, True]:
            # Cannot test
            for dim in self.dims:  # in [d for d in self.dims if not isinstance(d, Tuple)]:
                err_msg = f"Error for :\n dim = {dim}\n keepdim = {keepdim}"

                tensor = torch.randn(
                    (7, 9, 11, 13)
                )  # Need odd number of values to properly compare torch to numpy

                # Prepare the arguments for numpy vs torch
                if dim is not None:
                    torch_kwargs = {"dim": dim, "keepdim": keepdim}
                    numpy_kwargs = {"axis": dim, "keepdims": keepdim}
                else:
                    torch_kwargs = {}
                    numpy_kwargs = {}

                # Compare the nan-median
                torch_med = nan_median(tensor, **torch_kwargs)
                numpy_med = np.nanmedian(tensor.numpy(), **numpy_kwargs)
                torch_sum = torch.nansum(tensor, **torch_kwargs)
                np.testing.assert_almost_equal(torch_med.numpy(), numpy_med, decimal=4, err_msg=err_msg)
                self.assertListEqual(list(torch_med.shape), list(torch_sum.shape))

    def test_nan_mad(self):
        for normal in [False, True]:
            # Cannot test
            for dim in self.dims:  # in [d for d in self.dims if not isinstance(d, Tuple)]:
                err_msg = f"Error for :\n dim = {dim}\n normal = {normal}"

                tensor = torch.randn(
                    (7, 9, 11, 13)
                )  # Need odd number of values to properly compare torch to numpy

                # Prepare the arguments for numpy vs torch
                if dim is not None:
                    torch_kwargs = {"dim": dim, "keepdim": False, "normal": normal}
                    numpy_kwargs = {"axis": dim, "nan_policy": "omit", "scale": 1 / 1.4826 if normal else 1.0}
                else:
                    torch_kwargs = {"normal": normal}
                    numpy_kwargs = {"axis": dim, "nan_policy": "omit", "scale": 1 / 1.4826 if normal else 1.0}

                # Compare the nan-median
                torch_mad = nan_mad(tensor, **torch_kwargs)
                numpy_mad = sp.stats.median_abs_deviation(tensor.numpy(), **numpy_kwargs)
                np.testing.assert_almost_equal(torch_mad.numpy(), numpy_mad, decimal=4, err_msg=err_msg)


class test_SafeRun(ut.TestCase):
    def test_safe_run(self):
        # Error is caught
        with SafeRun(name="bob", raise_error=False, verbose=0):
            raise ValueError("This is an error")

        # Error is caught
        with SafeRun(name="bob", raise_error=False, verbose=0):
            2 + None

        # Error is not caught
        with self.assertRaises(ValueError):
            with SafeRun(name="bob", raise_error=True, verbose=0):
                raise ValueError("This is an error")

        # Error is not caught
        with self.assertRaises(TypeError):
            with SafeRun(name="bob", raise_error=True, verbose=0):
                2 + None

        # No error. Runs correctly
        with SafeRun(name="bob", raise_error=True, verbose=0):
            print("This is not an error")

        # No error. Runs correctly
        with SafeRun(name="bob", raise_error=False, verbose=0):
            print("This is not an error")


if __name__ == "__main__":
    ut.main()
