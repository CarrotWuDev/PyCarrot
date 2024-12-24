import numpy as np
from ..nn.module import Module
from ..nn.parameter import Parameter

class Optim(Module):
    def zero_grad(self):
        for parameter in self.parameters:
            parameter: Parameter
            parameter.zero_grad()