import os
import torch
import numpy as np
from matplotlib import pyplot as plt

from rdkit.Chem import AllChem


def save_im(im_dir, im_name, ext=["svg", "png"], dpi=600):

    if not os.path.exists(im_dir):
        if im_dir[-1] not in ["/", "\\"]:
            im_dir += "/"
        os.makedirs(im_dir)

    if isinstance(ext, str):
        ext = [ext]

    full_name = os.path.join(im_dir, im_name)
    for this_ext in ext:
        plt.savefig(f"{full_name}.{this_ext}", dpi=dpi, bbox_inches="tight", pad_inches=0)


def to_tensor(x, device=None, dtype=None):
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
    elif isinstance(x, torch.Tensor):
        pass
    else:
        x = torch.tensor(x)

    x = x.to(dtype=dtype, device=device)

    return x


def is_dtype_torch_tensor(dtype):
    r"""
    Verify if the dtype is a torch dtype

    Parameters:
        dtype: dtype
            The dtype of a value. E.g. np.int32, str, torch.float

    Returns:
        A boolean saying if the dtype is a torch dtype
    """
    return isinstance(dtype, torch.dtype) or (dtype == torch.Tensor)


def is_dtype_numpy_array(dtype):
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


def one_of_k_encoding(val, num_classes, dtype=int):
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


def is_device_cuda(device, ignore_errors=False):
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
    def __init__(self, dim=-1):
        super().__init__()
        self.dim = dim

    def forward(self, *args, **kwargs):
        h = []
        for module in self:
            h.append(module.forward(*args, **kwargs))

        return torch.cat(h, dim=self.dim)
