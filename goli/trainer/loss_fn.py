import torch
from torch import Tensor

class BCELossBalanced(torch.nn.BCELoss):
    
    def forward(self, input: Tensor, target:Tensor) -> Tensor:
        assert self.weight is None

        dims = list(range(input.ndim - 1))


