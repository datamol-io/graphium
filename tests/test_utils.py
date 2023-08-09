"""
Unit tests for the metrics and wrappers of graphium/utils/...
"""

import torch
import numpy as np
import scipy as sp
import unittest as ut
import gzip

from graphium.utils.read_file import file_opener
from graphium.utils.tensor import (
    nan_mad,
    nan_mean,
    nan_std,
    nan_var,
    nan_median,
    dict_tensor_fp16_to_fp32,
    tensor_fp16_to_fp32,
)
from graphium.utils.safe_run import SafeRun


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


def test_file_opener(tmp_path):
    # Create a temporary file
    txt_file = tmp_path / "test.txt"
    txt_file.write_text("Hello, World!")

    # Test opening file in read mode
    with file_opener(txt_file, "r") as f:
        assert f.read() == "Hello, World!"

    # Test opening file in write mode
    with file_opener(txt_file, "w") as f:
        f.write("New text")

    with file_opener(txt_file, "r") as f:
        assert f.read() == "New text"

    # Create a temporary gzip file
    gzip_file = tmp_path / "test.txt.gz"
    with gzip.open(gzip_file, "wt") as f:
        f.write("Hello, Gzip!")

    # Test opening gzip file in read mode
    with file_opener(gzip_file, "r") as f:
        assert f.read() == "Hello, Gzip!"


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


class TestTensorFp16ToFp32(ut.TestCase):
    def test_tensor_fp16_to_fp32(self):
        # Create a tensor
        tensor = torch.randn(10, 10).half()

        # Convert the tensor to fp32
        tensor_fp32 = tensor_fp16_to_fp32(tensor)
        self.assertTrue(tensor_fp32.dtype == torch.float32)

        # Don't convert the tensor to fp32
        tensor = torch.randn(10, 10).int()
        tensor_fp32 = tensor_fp16_to_fp32(tensor)
        self.assertFalse(tensor_fp32.dtype == torch.float32)

        # Don't convert the tensor to fp32
        tensor = torch.randn(10, 10).double()
        tensor_fp32 = tensor_fp16_to_fp32(tensor)
        self.assertFalse(tensor_fp32.dtype == torch.float32)

    def test_dict_tensor_fp16_to_fp32(self):
        # Create a dictionary of tensors
        tensor_dict = {
            "a": torch.randn(10, 10).half(),
            "b": torch.randn(10, 10).half(),
            "c": torch.randn(10, 10).double(),
            "d": torch.randn(10, 10).half(),
            "e": torch.randn(10, 10).float(),
            "f": torch.randn(10, 10).half(),
            "g": torch.randn(10, 10).int(),
            "h": {
                "h1": torch.randn(10, 10).double(),
                "h2": torch.randn(10, 10).half(),
                "h3": torch.randn(10, 10).float(),
                "h4": torch.randn(10, 10).half(),
                "h5": torch.randn(10, 10).int(),
            },
        }

        # Convert the dictionary to fp32
        tensor_dict_fp32 = dict_tensor_fp16_to_fp32(tensor_dict)

        # Check that the dictionary is correctly converted
        for key, tensor in tensor_dict_fp32.items():
            if key in ["a", "b", "d", "e", "f"]:
                self.assertEqual(tensor.dtype, torch.float32)
            elif key in ["h"]:
                for key2, tensor2 in tensor.items():
                    if key2 in ["h2", "h3", "h4"]:
                        self.assertEqual(tensor2.dtype, torch.float32)
                    else:
                        self.assertNotEqual(tensor2.dtype, torch.float32)
            else:
                self.assertNotEqual(tensor.dtype, torch.float32)


if __name__ == "__main__":
    ut.main()
