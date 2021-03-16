from typing import Union, List, Optional, Any, Tuple, Callable

import torch
import numpy as np

from torch.utils.data import Dataset


class BaseDataset(Dataset):
    r"""
    A base class to use to define the different datasets
    """

    def __init__(
        self,
        feats: List[Any],
        labels: Union[List[float], List[int], np.ndarray],
        weights: Optional[List[float]] = None,
        collate_fn: Optional[Callable] = None,
    ):
        r"""
        Parameters:
            feats:
                List of input features
            labels:
                iterable over the targets associated to the input features
            weights:
                sample or prediction weights to be used in a cost function
            collate_fn:
                The collate function used to batch multiple input features.
                If None, the default collate function is used.

        """

        super().__init__()

        # simple attributes
        self.feats = feats
        self.labels = torch.as_tensor(labels)
        self.weights = weights
        if self.weights is not None:
            self.weights = torch.as_tensor(self.weights)
        self._collate_fn = collate_fn

        # Assert good lengths
        if len(self.feats) != len(self.labels):
            raise ValueError(
                f"`feats` and `labels` must be the same length. Found {len(self.feats)}, {len(self.labels)}"
            )
        if (self.weights is not None) and (len(self.weights) != len(self.labels)):
            raise ValueError(
                f"`weights` and `labels` must be the same length. Found {len(self.weights)}, {len(self.labels)}"
            )

    @property
    def collate_fn(self) -> Optional[Callable]:
        r"""
        Returns the collate function of the current Dataset
        """
        return self._collate_fn

    def __len__(self) -> int:
        r"""
        Returns the number of elements of the current Dataset
        """
        return len(self.feats)

    def __getitem__(self, idx: int) -> dict:
        r"""
        Returns the input featues, the labels, and the weights at the
        specified index.

        Returns:
            feats: The input features at index `idx`

            labels: Th labels at index `idx`

            weights: Th weights at index `idx` if weights are provided. Otherwise
                it is ignored.

        """
        datum = {}
        datum["features"] = self.feats[idx]
        datum["labels"] = self.labels[idx]

        if self.weights is not None:
            datum["weights"] = self.weights[idx]

        return datum


class SmilesDataset(BaseDataset):

    """
    A simple dataset for working with Smiles data and apply any
    kind of smiles transformer to it, such as a transformer that
    returns a `DGL.graph` object.
    """

    def __init__(
        self,
        smiles: List[str],
        labels: List[Union[float, int]],
        weights: Optional[List[float]] = None,
        smiles_transform: Optional[Callable] = None,
        collate_fn: Optional[Callable] = None,
    ):
        """
        Parameters:
            smiles: iterable of smiles molecules
            labels:
                iterable over the targets associated to the input features
            weights:
                sample or prediction weights to be used in a cost function
            smiles_transform:
                Feature Transformation to apply over the smiles,
                for example fingerprints transformers or dgl graphs transformers.
            collate_fn:
                The collate function used to batch multiple input features.
                If None, the default collate function is used.
        """

        self.smiles = smiles
        self.smiles_transform = smiles_transform

        if self.smiles_transform is not None:
            feats = [self.smiles_transform(s) for s in self.smiles]
        else:
            feats = self.smiles

        super().__init__(
            feats=feats,
            labels=labels,
            weights=weights,
            collate_fn=collate_fn,
        )

    def __getitem__(self, idx: int) -> dict:
        r"""
        Returns the input featues, the labels, and the weights at the
        specified index.

        Returns:
            feats: The input features at index `idx`

            labels: Th labels at index `idx`

            weights: Th weights at index `idx` if weights are provided. Otherwise
                it is ignored.

        """
        datum = super().__getitem__(idx)

        # NOTE(hadim): we could make this optional. It's often convenient
        # to have the smiles alongside your features and labels.
        datum["smiles"] = self.smiles[idx]

        return datum
