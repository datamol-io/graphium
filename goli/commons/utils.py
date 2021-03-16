import os
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from typing import List, Union
from inspect import getfullargspec
from copy import copy
from loguru import logger

from rdkit.Chem import AllChem


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


def to_tensor(
    x: Union[np.ndarray, torch.Tensor, pd.DataFrame],
    device: Union[torch.device, type(None)] = None,
    dtype: Union[torch.dtype, type(None)] = None,
) -> torch.Tensor:
    r"""
    Convert a numpy array to tensor. The tensor type will be
    the same as the original array, unless specify otherwise

    Parameters:
        x: numpy.ndarray
            Numpy array to convert to tensor type
        device: torch.device
        dtype: torch.dtype
            Enforces new data type for the output

    Returns:
        New torch.Tensor

    """
    if isinstance(x, torch.Tensor):
        pass
    elif isinstance(x, np.ndarray):
        x = torch.from_numpy(x)
    elif isinstance(x, pd.DataFrame):
        x = torch.from_numpy(x.values)
    elif isinstance(x, torch.Tensor):
        pass
    else:
        x = torch.tensor(x)

    x = x.to(dtype=dtype, device=device)

    return x


def is_dtype_torch_tensor(dtype: Union[np.dtype, torch.dtype]) -> bool:
    r"""
    Verify if the dtype is a torch dtype

    Parameters:
        dtype: dtype
            The dtype of a value. E.g. np.int32, str, torch.float

    Returns:
        A boolean saying if the dtype is a torch dtype
    """
    return isinstance(dtype, torch.dtype) or (dtype == torch.Tensor)


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


def one_of_k_encoding(val: int, num_classes: int, dtype=int) -> np.ndarray:
    r"""Converts a single value to a one-hot vector.

    Parameters:
        val: int
            class to be converted into a one hot vector
            (integers from 0 to num_classes).
        num_classes: iterator
            a list or 1D array of allowed
            choices for val to take
        dtype: type
            data type of the the return.
            Possible types are int, float, bool, ...
    Returns:
        A numpy 1D array of length len(num_classes) + 1
    """

    encoding = np.zeros(len(num_classes) + 1, dtype=dtype)
    # not using index of, in case, someone fuck up
    # and there are duplicates in the allowed choices
    for i, v in enumerate(num_classes):
        if v == val:
            encoding[i] = 1
    if np.sum(encoding) == 0:  # aka not found
        encoding[-1] = 1
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


class ModuleListConcat(torch.nn.ModuleList):
    def __init__(self, dim: int = -1):
        super().__init__()
        self.dim = dim

    def forward(self, *args, **kwargs) -> torch.Tensor:
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
