from typing import Optional
from loguru import logger
import numpy as np
import torch
from torch import Tensor


class LabelNormalization:
    def __init__(
        self,
        method: Optional[str] = None,
        min_clipping: Optional[int] = None,
        max_clipping: Optional[int] = None,
        normalize_val_test: Optional[bool] = False,
        verbose: Optional[bool] = True,
    ):
        """
        Parameters:
        method: str
            Normalization method. Supports the following values:
            - `None` (default): No normalization applied
            - `normal`: Normalize to have 0-mean and 1-variance
            - `unit`: Normalize to have all values in the range 0-1

        min_clipping: int
            Minimum value to clip to. If `None` (default), no clipping is applied.
            For example, if `min_clipping` is -2, all values below -2 will be clipped to -2.
            This is applied before the normalization.

        max_clipping: int
            Maximum value to clip to. If `None` (default), no clipping is applied.
            For example, if `max_clipping` is 2, all values above 2 will be clipped to 2.
            This is applied before the normalization.
        """

        self.method = method
        self.min_clipping = min_clipping
        self.max_clipping = max_clipping
        self.normalize_val_test = normalize_val_test
        self.verbose = verbose
        self.data_max = None
        self.data_min = None
        self.data_mean = None
        self.data_std = None

    def calculate_statistics(self, array):
        """
        Saves the normalization parameters (e.g. mean and variance) to the object.
        """
        # add axis = 0 to make sure that the statistics for multiple column labels is a vector or list instead of a scalar
        self.data_max = np.nanmax(array, axis=0).tolist()
        self.data_min = np.nanmin(array, axis=0).tolist()
        self.data_mean = np.nanmean(array, axis=0).tolist()  # 5.380503871833475 for pcqm4mv2
        self.data_std = np.nanstd(array, axis=0).tolist()  # 1.17850688410978995 for pcqm4mv2
        if self.verbose:
            logger.info(f"Max value for normalization '{self.data_max}'")
            logger.info(f"Min value for normalization '{self.data_min}'")
            logger.info(f"Mean value for normalization '{self.data_mean}'")
            logger.info(f"STD value for normalization '{self.data_std}'")

    def normalize(self, input):
        """
        Apply the normalization method to the data.
        Saves the normalization parameters (e.g. mean and variance) to the object.
        """
        assert self.data_max is not None, "calculate_statistic must be called before applying normalization"
        if self.min_clipping is not None:
            self.data_min = max(self.min_clipping, self.data_min)
        if self.max_clipping is not None:
            self.data_max = min(self.max_clipping, self.data_max)
        clipping = self.min_clipping is not None and self.max_clipping is not None
        # Need to check since np.clip fails if both a_min and a_max are None
        if clipping:
            if isinstance(input, np.ndarray):
                input = np.clip(input, a_min=self.data_min, a_max=self.data_max)
            elif isinstance(input, Tensor):
                input = torch.clip(input, min=self.data_min, max=self.data_max)
        if self.method is None:
            return input
        elif self.method == "normal":
            return (input - self.data_mean) / self.data_std
        elif self.method == "unit":
            return (input - self.data_min) / (self.data_max - self.data_min)
        else:
            raise ValueError(f"normalization method {self.method} not recognised.")

    def denormalize(self, input):
        """
        Apply the inverse of the normalization method to the data.
        """
        if self.method is None:
            return input
        elif self.method == "normal":
            mean, std = torch.tensor(self.data_mean), torch.tensor(self.data_std)
            if input.device.type != "ipu":  # Cast to device if not on IPU
                mean, std = mean.to(input.device), std.to(input.device)
            return (input * std) + mean
        elif self.method == "unit":
            dmax, dmin = torch.tensor(self.data_max), torch.tensor(self.data_min)
            if input.device.type != "ipu":  # Cast to device if not on IPU
                dmax, dmin = dmax.to(input.device), dmin.to(input.device)
            return input * (dmax - dmin) + dmin
        else:
            raise ValueError(f"normalization method {self.method} not recognised.")
