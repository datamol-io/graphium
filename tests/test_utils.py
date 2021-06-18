"""
Unit tests for the metrics and wrappers of goli/utils/...
"""

from goli.utils.tensor import nan_mean, nan_std, nan_var
import torch
import numpy as np
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


if __name__ == "__main__":
    ut.main()
