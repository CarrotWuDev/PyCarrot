from ..carrot import Carrot
from .parameter import Parameter
import numpy as np
from .module import Module

class Linear(Module):
    def __init__(self, input_dimen, output_dimen) -> None:
        super().__init__()
        self.weight = Parameter(np.random.normal(0, 1, [input_dimen, output_dimen]), requires_grad=True)
        self.bias = Parameter(np.zeros([output_dimen]), requires_grad=True)
        pass

    def forward(self, input:Carrot) -> Carrot:
        output = input @ self.weight + self.bias
        return output

if __name__ == "__main__":
    linear = Linear(2, 1)
    print(linear.state_dict())