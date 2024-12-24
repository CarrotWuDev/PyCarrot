import numpy as np
from ..carrot import Carrot


class Parameter(Carrot):
    def __init__(
        self, data: np.ndarray | Carrot, requires_grad=True, child_nodes=[], name=""
    ) -> None:
        if type(data) == np.ndarray:
            super().__init__(data, requires_grad, child_nodes, name)
            pass
        else:
            super().__init__(data.data, requires_grad, child_nodes, name)
            pass
        pass
