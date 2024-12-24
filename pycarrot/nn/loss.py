from ..carrot import Carrot
import numpy as np
from .parameter import Parameter
from .module import Module
from .functional import mse_loss, cross_entropy_loss


class MSELoss(Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, input: Carrot, target: Carrot) -> Carrot:
        output = mse_loss(input, target)
        return output


class CrossEntropyLoss(Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, input: Carrot, target: Carrot) -> Carrot:
        output = cross_entropy_loss(input, target)
        return output
