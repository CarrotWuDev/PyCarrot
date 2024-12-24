from ..carrot import Carrot
import numpy as np
from .parameter import Parameter
from .module import Module
from .functional import relu, tanh, sigmoid, softmax


class ReLU(Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, input: Carrot) -> Carrot:
        output = relu(input)
        return output


class Sigmoid(Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, input: Carrot) -> Carrot:
        output = sigmoid(input)
        return output


class Tanh(Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, input: Carrot) -> Carrot:
        output = tanh(input)
        return output


class Softmax(Module):
    def __init__(self) -> None:
        super().__init__()
        pass

    def forward(self, input: Carrot) -> Carrot:
        output = softmax(input)
        return output
