import os
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import Iterable, List, Union, Any, Callable, Dict
from inspect import getfullargspec
from copy import copy, deepcopy
from loguru import logger

from rdkit.Chem import AllChem

import torch
from torch import Tensor


def save_im(im_dir, im_name: str, ext: List[str] = ["svg", "png"], dpi: int = 600) -> None:
    if not os.path.exists(im_dir):
        if im_dir[-1] not in ["/", "\\"]:
            im_dir += "/"
        os.makedirs(im_dir)

    if isinstance(ext, str):
        ext = [ext]

    full_name = os.path.join(im_dir, im_name)
    for this_ext in ext:
        plt.savefig(f"{full_name}.{this_ext}", dpi=dpi, bbox_inches="tight", pad_inches=0)


def is_dtype_torch_tensor(dtype: Union[np.dtype, torch.dtype]) -> bool:
    r"""
    Verify if the dtype is a torch dtype

    Parameters:
        dtype: dtype
            The dtype of a value. E.g. np.int32, str, torch.float

    Returns:
        A boolean saying if the dtype is a torch dtype
    """
    return isinstance(dtype, torch.dtype) or (dtype == Tensor)


def is_dtype_numpy_array(dtype: Union[np.dtype, torch.dtype]) -> bool:
    r"""
    Verify if the dtype is a numpy dtype

    Parameters:
        dtype: dtype
            The dtype of a value. E.g. np.int32, str, torch.float

    Returns:
        A boolean saying if the dtype is a numpy dtype
    """
    is_torch = is_dtype_torch_tensor(dtype)
    is_num = dtype in (int, float, complex)
    if hasattr(dtype, "__module__"):
        is_numpy = dtype.__module__ == "numpy"
    else:
        is_numpy = False

    return (is_num or is_numpy) and not is_torch


def one_of_k_encoding(val: Any, classes: Iterable[Any]) -> List[int]:
    r"""Converts a single value to a one-hot vector.

    Parameters:
        val: int
            class to be converted into a one hot vector
            (integers from 0 to num_classes).
        num_classes: iterator
            a list or 1D array of allowed
            choices for val to take
    Returns:
        A list of length len(num_classes) + 1
    """
    encoding = [False] * (len(classes) + 1)
    found = False
    for i, v in enumerate(classes):
        if v == val:
            encoding[i] = True
            found = True
            break
    if not found:
        encoding[-1] = True
    return encoding


def is_device_cuda(device: torch.device, ignore_errors: bool = False) -> bool:
    r"""Check wheter the given device is a cuda device.

    Parameters:
        device: str, torch.device
            object to check for cuda
        ignore_errors: bool
            Whether to ignore the error if the device is not recognized.
            Otherwise, ``False`` is returned in case of errors.
    Returns:
        is_cuda: bool
    """

    if ignore_errors:
        is_cuda = False
        try:
            is_cuda = torch.device(device).type == "cuda"
        except:
            pass
    else:
        is_cuda = torch.device(device).type == "cuda"
    return is_cuda


def nan_mean(input: Tensor, *args, **kwargs) -> Tensor:
    r"""
    Return the mean of all elements, while ignoring the NaNs.

    Parameters:

        input: The input tensor.

        dim (int or tuple(int)): The dimension or dimensions to reduce.

        keepdim (bool): whether the output tensor has dim retained or not.

        dtype (torch.dtype, optional):
            The desired data type of returned tensor.
            If specified, the input tensor is casted to dtype before the operation is performed.
            This is useful for preventing data type overflows. Default: None.

    Returns:
        output: The resulting mean of the tensor
    """

    sum = torch.nansum(input, *args, **kwargs)
    num = torch.sum(~torch.isnan(input), *args, **kwargs)
    mean = sum / num
    return mean


def nan_median(input: Tensor, **kwargs) -> Tensor:
    r"""
    Return the median of all elements, while ignoring the NaNs.
    Contrarily to `torch.nanmedian`, this function supports a list
    of dimensions, or `dim=None`, and does not return the index of the median

    Parameters:

        input: The input tensor.

        dim (int or tuple(int)): The dimension or dimensions to reduce.

        keepdim (bool): whether the output tensor has dim retained or not.

        dtype (torch.dtype, optional):
            The desired data type of returned tensor.
            If specified, the input tensor is casted to dtype before the operation is performed.
            This is useful for preventing data type overflows. Default: None.

    Returns:
        output: The resulting median of the tensor.
            Contrarily to `torch.median`, it does not return the index of the median
    """

    dim = kwargs.pop("dim", None)
    keepdim = kwargs.pop("keepdim", False)

    if isinstance(dim, Iterable) and not isinstance(dim, str):
        dim = list(dim)
        dim.sort()
        # Implement the median for a list of dimensions
        for d in dim:
            input = input.unsqueeze(-1)
            input = input.transpose(d, -1)
        input = input.flatten(-len(dim))
        median, _ = torch.nanmedian(input, dim=-1, keepdim=False)
        if not keepdim:
            for d in dim[::-1]:
                median = median.squeeze(d)
    else:
        if dim is None:
            median = torch.nanmedian(input.flatten().float()).to(input.dtype)
        else:
            median, _ = torch.nanmedian(input, dim=dim, keepdim=keepdim)

    return median


def nan_var(input: Tensor, unbiased: bool = True, **kwargs) -> Tensor:
    r"""
    Return the variace of all elements, while ignoring the NaNs.
    If unbiased is True, Bessel’s correction will be used.
    Otherwise, the sample deviation is calculated, without any correction.

    Parameters:

        input: The input tensor.

        unbiased: whether to use Bessel’s correction (δN=1\delta N = 1δN=1).

        dim (int or tuple(int)): The dimension or dimensions to reduce.

        keepdim (bool): whether the output tensor has dim retained or not.

        dtype (torch.dtype, optional):
            The desired data type of returned tensor.
            If specified, the input tensor is casted to dtype before the operation is performed.
            This is useful for preventing data type overflows. Default: None.

    Returns:
        output: The resulting variance of the tensor
    """

    mean_kwargs = deepcopy(kwargs)
    mean_kwargs.pop("keepdim", None)
    dim = mean_kwargs.pop("dim", [ii for ii in range(input.ndim)])
    mean = nan_mean(input, dim=dim, keepdim=True, **mean_kwargs)
    dist = input - mean
    dist2 = dist * dist
    var = nan_mean(dist2, **kwargs)

    if unbiased:
        num = torch.sum(~torch.isnan(input), **kwargs)
        var = var * num / (num - 1)

    return var


def nan_std(input: Tensor, unbiased: bool = True, **kwargs) -> Tensor:
    r"""
    Return the standard deviation of all elements, while ignoring the NaNs.
    If unbiased is True, Bessel’s correction will be used.
    Otherwise, the sample deviation is calculated, without any correction.

    Parameters:

        input: The input tensor.

        unbiased: whether to use Bessel’s correction (δN=1\delta N = 1δN=1).

        dim (int or tuple(int)): The dimension or dimensions to reduce.

        keepdim (bool): whether the output tensor has dim retained or not.

        dtype (torch.dtype, optional):
            The desired data type of returned tensor.
            If specified, the input tensor is casted to dtype before the operation is performed.
            This is useful for preventing data type overflows. Default: None.

    Returns:
        output: The resulting standard deviation of the tensor
    """
    variances = nan_var(input=input, unbiased=unbiased, **kwargs)
    return torch.sqrt(variances.float()).to(variances.dtype)


def nan_mad(input: Tensor, normal: bool = True, **kwargs) -> Tensor:
    r"""
    Return the median absolute deviation of all elements, while ignoring the NaNs.

    Parameters:

        input: The input tensor.

        normal: whether to multiply the result by 1.4826 to mimic the
            standard deviation for normal distributions.

        dim (int or tuple(int)): The dimension or dimensions to reduce.

        keepdim (bool): whether the output tensor has dim retained or not.

        dtype (torch.dtype, optional):
            The desired data type of returned tensor.
            If specified, the input tensor is casted to dtype before the operation is performed.
            This is useful for preventing data type overflows. Default: None.

    Returns:
        output: The resulting median absolute deviation of the tensor
    """
    median_kwargs = deepcopy(kwargs)
    median_kwargs.pop("keepdim", None)
    dim = median_kwargs.pop("dim", [ii for ii in range(input.ndim)])
    median = nan_median(input, dim=dim, keepdim=True, **median_kwargs)
    dist = (input - median).abs()
    mad = nan_median(dist, **kwargs)
    if normal:
        mad = mad * 1.4826
    return mad


class ModuleWrap(torch.nn.Module):
    r"""
    Wrap a function into a `torch.nn.Module`, with possible `*args` and `**kwargs`

    Parameters:
        func: function to wrap into a module
    """

    def __init__(self, func, *args, **kwargs) -> None:
        super().__init__()
        self.func = func
        self.__name__ = f"ModuleWrap({self.func.__name__})"
        self.args = args
        self.kwargs = kwargs

    def forward(self, *args, **kwargs):
        """
        Calls the function `self.func` with the arguments `self.func(*self.args, *args, **self.kwargs, **kwargs)`
        """
        return self.func(*self.args, *args, **self.kwargs, **kwargs)

    def __repr__(self):
        return self.__name__


class ModuleListConcat(torch.nn.ModuleList):
    r"""
    A list of neural modules similar to `torch.nn.ModuleList`,
    but where the modules are applied on the same input and
    concatenated together, instead of being applied sequentially.

    Parameters:
        dim: The dimension for the concatenation
    """

    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, *args, **kwargs) -> Tensor:
        r"""
        Apply all layers on the `args` and `kwargs`, and concatenate
        their output alongside the dimension `self.dim`.
        """
        h = []
        for module in self:
            h.append(module.forward(*args, **kwargs))

        return torch.cat(h, dim=self.dim)


def parse_valid_args(param_dict, fn):
    r"""
    Check if a function takes the given argument.

    Parameters
    ----------
    fn: func
        The function to check the argument.
    param_dict: dict
        Dictionary of the argument.

    Returns
    -------
        param_dict: dict
            Valid paramter dictionary for the given fucntions.
    """
    param_dict_cp = copy(param_dict)
    for key, param in param_dict.items():
        if not arg_in_func(fn=fn, arg=key):
            logger.warning(
                f"{key} is not an available argument for {fn.__name__}, and is ignored by default."
            )
            param_dict_cp.pop(key)

    return param_dict_cp


def arg_in_func(fn, arg):
    r"""
    Check if a function takes the given argument.

    Parameters
    ----------
    fn: func
        The function to check the argument.
    arg: str
        The name of the argument.

    Returns
    -------
        res: bool
            True if the function contains the argument, otherwise False.
    """
    fn_args = getfullargspec(fn)
    return (fn_args.varkw is not None) or (arg in fn_args[0])


def tensor_fp16_to_fp32(tensor: Tensor) -> Tensor:
    r"""Cast a tensor from fp16 to fp32 if it is in fp16

    Parameters:
        tensor: A tensor. If it is in fp16, it will be casted to fp32

    Returns:
        tensor: The tensor casted to fp32 if it was in fp16
    """
    if tensor.dtype == torch.float16:
        return tensor.to(dtype=torch.float32)
    return tensor


def dict_tensor_fp16_to_fp32(
    dict_tensor: Union[Tensor, Dict[str, Tensor], Dict[str, Dict[str, Tensor]]]
) -> Union[Tensor, Dict[str, Tensor], Dict[str, Dict[str, Tensor]]]:
    r"""Recursively Cast a dictionary of tensors from fp16 to fp32 if it is in fp16

    Parameters:
        dict_tensor: A recursive dictionary of tensors. To be casted to fp32 if it was in fp16.

    Returns:
        dict_tensor: The recursive dictionary of tensors casted to fp32 if it was in fp16
    """
    if isinstance(dict_tensor, dict):
        for key, value in dict_tensor.items():
            dict_tensor[key] = dict_tensor_fp16_to_fp32(value)
    else:
        dict_tensor = tensor_fp16_to_fp32(dict_tensor)

    return dict_tensor
