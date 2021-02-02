
import abc
import torch
import torch.nn as nn
from goli.dgl.base_layers import FCLayer


RESIDUAL_TYPE_DICT = {
    "none": ResidualConnectionNone,
    "simple": ResidualConnectionSimple,
    "weighted": ResidualConnectionWeighted,
    "concat": ResidualConnectionConcat,
    "densenet": ResidualConnectionDenseNet,
}


class ResidualConnectionBase(nn.Module):

    def __init__(self, skip_steps=1):
        super().__init__()
        self.skip_steps = skip_steps


    def _bool_apply_skip_step(self, step_idx):
        return (self.skip_steps != 0) and ((step_idx % self.skip_steps) == 1)

    @staticmethod
    @abc.abstractmethod
    def h_dim_increase_factor():
        ...

    @staticmethod
    @abc.abstractmethod
    def has_weights():
        ...


class ResidualConnectionNone(ResidualConnectionBase):
    def __init__(self, skip_steps=1):
        super().__init__(skip_steps=skip_steps)
    
    @staticmethod
    def h_dim_increase_factor():
        # Return `None`, "previous" or "cumulative"
        return None
    
    @staticmethod
    def has_weights():
        return False

    def forward(self, h, h_prev, step_idx):
        return h, h_prev


class ResidualConnectionSimple(ResidualConnectionBase):
    def __init__(self, skip_steps=1):
        super().__init__(skip_steps=skip_steps)
    
    @staticmethod
    def h_dim_increase_factor():
        return None
    
    @staticmethod
    def has_weights():
        return False

    def forward(self, h, h_prev, step_idx):

        if self._bool_apply_skip_step(step_idx):
            if step_idx > 1:
                h = h + h_prev
            h_prev = h
        
        return h, h_prev


class ResidualConnectionWeighted(ResidualConnectionBase):
    def __init__(self, depth, full_dims, skip_steps=1, dropout=0., 
                activation='relu', batch_norm=False, bias=False):
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
    def h_dim_increase_factor():
        return None
    
    @staticmethod
    def has_weights():
        return True

    def forward(self, h, h_prev, step_idx):

        if self._bool_apply_skip_step(step_idx):
            if step_idx > 1:
                h = h + self.residual_list[self.skip_count].forward(h_prev)
                self.skip_count += 1
            h_prev = h

        return h, h_prev


class ResidualConnectionConcat(ResidualConnectionBase):
    def __init__(self, skip_steps=1):
        super().__init__(skip_steps=skip_steps)

    
    @staticmethod
    def h_dim_increase_factor():
        return "previous"
    
    @staticmethod
    def has_weights():
        return False

    def forward(self, h, h_prev, step_idx):

        if self._bool_apply_skip_step(step_idx):
            h_in = h
            if step_idx > 1:
                h = torch.cat([h, h_prev], dim=-1)
            h_prev = h_in
        
        return h, h_prev



class ResidualConnectionDenseNet(ResidualConnectionBase):
    def __init__(self, skip_steps=1):
        super().__init__(skip_steps=skip_steps)

    
    @staticmethod
    def h_dim_increase_factor():
        return "cumulative"
    
    @staticmethod
    def has_weights():
        return False

    def forward(self, h, h_prev, step_idx):

        if self._bool_apply_skip_step(step_idx):
            if step_idx > 1:
                h = torch.cat([h, h_prev], dim=-1)
            h_prev = h
        
        return h, h_prev


