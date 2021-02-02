
import abc
import torch
import torch.nn as nn
from typing import List

from goli.dgl.base_layers import FCLayer


RESIDUAL_TYPE_DICT = {
    "none": ResidualConnectionNone,
    "simple": ResidualConnectionSimple,
    "weighted": ResidualConnectionWeighted,
    "concat": ResidualConnectionConcat,
    "densenet": ResidualConnectionDenseNet,
}


class ResidualConnectionBase(nn.Module):
    r"""
    Abstract class for the residual connections. Using this class,
    we implement different types of residual connections, such as
    the ResNet, weighted-ResNet, skip-concat and DensNet. 

    The following methods must be implemented in a children class

    - ``h_dim_increase_type()``

    - ``has_weights()``

    Parameters
    -------------

    skip_steps: int, Default=1
        The number of steps to skip between the residual connections.
    """

    def __init__(self, skip_steps: int=1):
        super().__init__()
        self.skip_steps = skip_steps


    def _bool_apply_skip_step(self, step_idx: int):
        r"""
        Whether to apply the skip connection, depending on the
        ``step_idx`` and ``self.skip_steps``.

        Parameters
        -------------

        step_idx: int
            The current layer step index.

        """
        return (self.skip_steps != 0) and ((step_idx % self.skip_steps) == 1)

    @staticmethod
    @abc.abstractmethod
    def h_dim_increase_type():
        r"""
        How does the dimension of the output features increases after each layer?

        Returns
        --------

        h_dim_increase_type: None or str
            - ``None``: The dimension of the output features do not change at each layer.
              E.g. ResNet.

            - "previous": The dimension of the output features is the concatenation of
              the previous layer with the new layer.

            - "cumulative": The dimension of the output features is the concatenation
              of all previous layers.

        """
        ...


    def get_true_out_dims(self, in_dims: List, out_dims: List):


        true_out_dims = [out_dims[0]]
        out_dims_at_skip = [0]
        for ii in range(1, len(in_dims)):
            
            # For the `None` type, don't change the output dims
            if self.h_dim_increase_type is None:
                true_out_dims.append(out_dims[ii])
            
            # For the "previous" type, add the previous layers when the skip connection applies
            elif self.h_dim_increase_type == "previous":
                if self._bool_apply_skip_step(step_idx=ii):
                    true_out_dims.append(out_dims[ii] + out_dims_at_skip[-1])
                    out_dims_at_skip.append(out_dims[ii])
                else:
                    true_out_dims.append(out_dims[ii])
            
            # For the "cumulative" type, add all previous layers when the skip connection applies
            elif self.h_dim_increase_type == "cumulative":
                if self._bool_apply_skip_step(step_idx=ii):
                    true_out_dims.append(out_dims[ii] + out_dims_at_skip[-1])
                    out_dims_at_skip.append(true_out_dims[ii])
                else:
                    true_out_dims.append(out_dims[ii])
            else:
                raise ValueError(f'undefined value: {self.h_dim_increase_type}')

        return true_out_dims


    @staticmethod
    @abc.abstractmethod
    def has_weights():
        r"""
        Returns
        --------

        has_weights: bool
            Whether the residual connection uses weights
        
        """
        ...


class ResidualConnectionNone(ResidualConnectionBase):
    r"""
    No residual connection. 
    This class is only used for simpler code compatibility
    """
    def __init__(self, skip_steps: int=1):
        super().__init__(skip_steps=skip_steps)
    
    @staticmethod
    def h_dim_increase_type():
        r"""
        Returns
        --------
        
        None: 
            The dimension of the output features do not change at each layer.
        """
        
        return None
    
    @staticmethod
    def has_weights():
        r"""
        Returns
        --------

        False
            The current class does not use weights
        
        """
        return False

    def forward(self, h: torch.Tensor, h_prev: torch.Tensor, step_idx: int):
        r"""
        Ignore the skip connection.

        Returns
        --------

        h: torch.Tensor(..., m)
            Return same as input.

        h_prev: torch.Tensor(..., m)
            Return same as input.

        """
        return h, h_prev


class ResidualConnectionSimple(ResidualConnectionBase):
    def __init__(self, skip_steps: int=1):
        super().__init__(skip_steps=skip_steps)
    
    @staticmethod
    def h_dim_increase_type():
        r"""
        Returns
        --------
        
        None: 
            The dimension of the output features do not change at each layer.
        """
        
        return None
    
    @staticmethod
    def has_weights():
        r"""
        Returns
        --------

        False
            The current class does not use weights
        
        """
        return False


    def forward(self, h: torch.Tensor, h_prev: torch.Tensor, step_idx: int):
        r"""
        Add ``h`` with the previous layers with skip connection ``h_prev``,
        similar to ResNet.

        Parameters
        -----------

        h: torch.Tensor(..., m)
            The current layer features

        h_prev: torch.Tensor(..., m), None
            The features from the previous layer with a skip connection.
            At ``step_idx==0``, ``h_prev`` can be set to ``None``. 

        step_idx: int
            Current layer index or step index in the forward loop of the architecture.

        Returns
        --------

        h: torch.Tensor(..., m)
            Either return ``h`` unchanged, or the sum with
            on ``h_prev``, depending on the ``step_idx`` and ``self.skip_steps``.

        h_prev: torch.Tensor(..., m)
            Either return ``h_prev`` unchanged, or the same value as ``h``, 
            depending on the ``step_idx`` and ``self.skip_steps``.

        """
        if self._bool_apply_skip_step(step_idx):
            if step_idx > 1:
                h = h + h_prev
            h_prev = h
        
        return h, h_prev


class ResidualConnectionWeighted(ResidualConnectionBase):
    def __init__(self, depth, full_dims, skip_steps: int=1, dropout=0., 
                activation='none', batch_norm=False, bias=False):
        super().__init__(skip_steps=skip_steps)
        
        self.residual_list = nn.ModuleList()
        self.skip_count = 0

        for ii in range(1, self.depth, self.skip_steps):
            this_dim = self.full_dims[ii]
            self.residual_list.append(
                nn.Sequential(
                    FCLayer(
                        this_dim,
                        this_dim,
                        activation=self.activation,
                        dropout=self.dropout,
                        batch_norm=self.batch_norm,
                        bias=False,
                    ),
                    nn.Linear(this_dim, this_dim, bias=False),
                )
            )

    
    @staticmethod
    def h_dim_increase_type():
        r"""
        Returns
        --------
        
        None: 
            The dimension of the output features do not change at each layer.
        """
        return None
    
    @staticmethod
    def has_weights():
        r"""
        Returns
        --------

        True
            The current class uses weights
        
        """
        return True
    

    def forward(self, h: torch.Tensor, h_prev: torch.Tensor, step_idx: int):
        r"""
        Add ``h`` with the previous layers with skip connection ``h_prev``, after
        a feed-forward layer.

        Parameters
        -----------

        h: torch.Tensor(..., m)
            The current layer features

        h_prev: torch.Tensor(..., m), None
            The features from the previous layer with a skip connection.
            At ``step_idx==0``, ``h_prev`` can be set to ``None``. 

        step_idx: int
            Current layer index or step index in the forward loop of the architecture.

        Returns
        --------

        h: torch.Tensor(..., m)
            Either return ``h`` unchanged, or the sum with the output of a NN layer
            on ``h_prev``, depending on the ``step_idx`` and ``self.skip_steps``.

        h_prev: torch.Tensor(..., m)
            Either return ``h_prev`` unchanged, or the same value as ``h``, 
            depending on the ``step_idx`` and ``self.skip_steps``.

        """

        if self._bool_apply_skip_step(step_idx):
            if step_idx > 1:
                h = h + self.residual_list[self.skip_count].forward(h_prev)
                self.skip_count += 1
            h_prev = h

        return h, h_prev


class ResidualConnectionConcat(ResidualConnectionBase):
    def __init__(self, skip_steps: int=1):
        super().__init__(skip_steps=skip_steps)


    @staticmethod
    def h_dim_increase_type():
        r"""
        Returns
        --------
        
        "previous": 
            The dimension of the output layer is the concatenation with the previous layer.
        """
        
        return "previous"
    
    @staticmethod
    def has_weights():
        r"""
        Returns
        --------

        False
            The current class does not use weights
        
        """
        return False


    def forward(self, h: torch.Tensor, h_prev: torch.Tensor, step_idx: int):
        r"""
        Concatenate ``h`` with the previous layers with skip connection ``h_prev``.

        Parameters
        -----------

        h: torch.Tensor(..., m)
            The current layer features

        h_prev: torch.Tensor(..., n), None
            The features from the previous layer with a skip connection.
            Usually, we have ``n`` equal to ``m``.
            At ``step_idx==0``, ``h_prev`` can be set to ``None``. 

        step_idx: int
            Current layer index or step index in the forward loop of the architecture.

        Returns
        --------

        h: torch.Tensor(..., m) or torch.Tensor(..., m + n)
            Either return ``h`` unchanged, or the concatenation
            with ``h_prev``, depending on the ``step_idx`` and ``self.skip_steps``.

        h_prev: torch.Tensor(..., m) or torch.Tensor(..., m + n)
            Either return ``h_prev`` unchanged, or the same value as ``h``, 
            depending on the ``step_idx`` and ``self.skip_steps``.

        """

        if self._bool_apply_skip_step(step_idx):
            h_in = h
            if step_idx > 1:
                h = torch.cat([h, h_prev], dim=-1)
            h_prev = h_in
        
        return h, h_prev



class ResidualConnectionDenseNet(ResidualConnectionBase):
    def __init__(self, skip_steps: int=1):
        super().__init__(skip_steps=skip_steps)

    
    @staticmethod
    def h_dim_increase_type():
        r"""
        Returns
        --------
        
        "cumulative": 
            The dimension of the output layer is the concatenation of all the previous layer.
        """
        
        return "cumulative"
    
    @staticmethod
    def has_weights():
        r"""
        Returns
        --------

        False
            The current class does not use weights
        
        """
        return False

    def forward(self, h: torch.Tensor, h_prev: torch.Tensor, step_idx: int):
        r"""
        Concatenate ``h`` with all the previous layers with skip connection ``h_prev``.

        Parameters
        -----------

        h: torch.Tensor(..., m)
            The current layer features

        h_prev: torch.Tensor(..., n), None
            The features from the previous layers.
            n = ((step_idx // self.skip_steps) + 1) * m

            At ``step_idx==0``, ``h_prev`` can be set to ``None``. 

        step_idx: int
            Current layer index or step index in the forward loop of the architecture.

        Returns
        --------

        h: torch.Tensor(..., m) or torch.Tensor(..., m + n)
            Either return ``h`` unchanged, or the concatenation
            with ``h_prev``, depending on the ``step_idx`` and ``self.skip_steps``.

        h_prev: torch.Tensor(..., m) or torch.Tensor(..., m + n)
            Either return ``h_prev`` unchanged, or the same value as ``h``, 
            depending on the ``step_idx`` and ``self.skip_steps``.

        """

        if self._bool_apply_skip_step(step_idx):
            if step_idx > 1:
                h = torch.cat([h, h_prev], dim=-1)
            h_prev = h
        
        return h, h_prev


