from .base import Optim
import numpy as np
from ..nn.parameter import Parameter


class SGD(Optim):
    def __init__(self, parameters, learning_rate=0.1, weight_decay=0.0) -> None:
        super().__init__()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.parameters = parameters
        pass

    def step(self):
        for parameter in self.parameters:
            parameter: Parameter
            decent_value = parameter.grad.data + self.weight_decay * parameter.data
            parameter.data -= self.learning_rate * decent_value
            pass
        pass
