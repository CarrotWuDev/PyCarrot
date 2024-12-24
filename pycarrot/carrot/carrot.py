"""
The carrot is the core of the this NN framework.
Carrot class object is the basic data type of nn framwork.
1. build Computational graphs
2. BP by Computational graphs 
"""

import numpy as np


# 1. Transform to ndarray
def data2ndarray(data):
    if isinstance(data, np.ndarray):
        return data
    else:
        return np.array(data)


# 2. Transform to Carrot(Tensor).
# It's operations among Carrots, so we must transform data to Carrot.
def data2carrot(data):
    if isinstance(data, Carrot):
        return data
    else:
        return Carrot(data)


# Define Carrot class like Tensor in PyTorch.
# We use numpy as carrot class's data structure.
# carrot._data is the ndarray type data.
class Carrot(object):
    def __init__(self, data, requires_grad=False, child_nodes=[], name=""):
        self._data = data2ndarray(data)
        self.requires_grad = requires_grad
        self.shape = self._data.shape
        self.grad = None
        self.child_nodes = child_nodes
        self.name = name
        # If the carrot objet is training, we need to prepare a grad Carrot.
        if self.requires_grad:
            self.zero_grad()

    @property
    def data(self) -> np.ndarray:
        return self._data

    @data.setter
    def data(self, new_data) -> None:
        self._data = data2ndarray(new_data)
        return None

    # Clear current carrot's grad to zero.
    def zero_grad(self) -> None:
        self.grad = Carrot(
            np.zeros_like(self.data, dtype=np.float64), requires_grad=False
        )
        return None

    def backward(self, grad: "Carrot" = None):
        if grad is None:
            if self.shape == ():
                grad = Carrot(1.0)
            else:
                grad = Carrot(np.ones(self.shape))
        # else: ?????
        self.grad.data += grad.data  # calculate current node grad
        for child_node, grad_wrt_func in self.child_nodes:
            child_node: Carrot
            child_node_grad = grad_wrt_func(
                grad.data
            )  # ????: why not prameter is self.grad.data ?
            child_node.backward(Carrot(child_node_grad))
        pass

    def __len__(self):
        """
        len: get carrot size.
        """
        return len(self._data)

    def __repr__(self):
        """
        get carrot official info for developer.
        """
        return f"Carrot:({self._data}, requires_grad={self.requires_grad})"

    def __getitem__(self, slice_index):
        """
        get carrot slice.
        """
        return _getitem(self, slice_index)

    def reshape(self, reshaped_size):
        """
        reshpe carrot size.
        """
        return _reshape(self, reshaped_size)

    def __neg__(self):
        """
        negative: - self
        """
        # return _mul(self, Carrot(-1))  # mul to neg, but error name
        return _neg(self)

    def __add__(self, other):
        """
        left add: self + other
        """
        return _add(self, data2carrot(other))

    def __radd__(self, other):
        """
        right add: other + self
        """
        return _add(data2carrot(other), self)

    def __mul__(self, other):
        """
        left mul: self * other
        """
        return _mul(self, data2carrot(other))

    def __rmul__(self, other):
        """
        right mul: other * self
        """
        return _mul(data2carrot(other), self)

    def __sub__(self, other):
        """
        left sub: self - other
        """
        # return _add(self, -data2carrot(other)) # add to sub, but error name
        return _sub(self, data2carrot(other))

    def __rsub__(self, other):
        """
        right sub: other - self
        """
        # return _add(-self, data2carrot(other)) # add to sub, but error name
        return _sub(data2carrot(other), self)

    def __pow__(self, n):
        """
        pow: self ** n
        """
        return _pow(self, n)

    def pow(self, n):
        """
        pow: self.pow(n)
        """
        return _pow(self, n)

    def __matmul__(self, other):
        """
        matrix left dot product: self @ other
        """
        return _matmul(self, data2carrot(other))

    def __rmatmul__(self, other):
        """
        matrix right dot product: other @ self
        """
        return _matmul(data2carrot(other), self)

    def mm(self, other):
        """
        matrix left dot product: self.mm(other)
        """
        return _matmul(self, data2carrot(other))

    def sum(self):
        """
        sum: sum carrot data.
        """
        return carrot_sum(self)

    def mean(self):
        """
        mean: calculate carrot data mean.
        """
        return carrot_mean(self)

    def clamp(self, min_value, max_value):
        """
        clamp: clip like numpy or tensor object.
        """
        return _clip(self, min_value, max_value)


def _add(left_operand: Carrot, right_operand: Carrot) -> Carrot:
    """
    add: left_operand + right_operand
    """
    result_data = left_operand.data + right_operand.data
    requires_grad = left_operand.requires_grad or right_operand.requires_grad
    child_nodes = []
    if left_operand.requires_grad:
        """
        We need to define the gradient function of the left operand.
        And the left operand's grad shape is the same as the left operand's shape.
        """

        def grad_wrt_left_operand(grad: np.ndarray) -> np.ndarray:
            current_operand_grad = grad
            dimen_gap = grad.ndim - left_operand.data.ndim
            for _ in range(dimen_gap):
                current_operand_grad = current_operand_grad.sum(axis=0)
            for i, dimen in enumerate(left_operand.shape):
                if dimen == 1:
                    current_operand_grad = current_operand_grad.sum(
                        axis=i, keepdims=True
                    )
            return current_operand_grad

        child_nodes.append((left_operand, grad_wrt_left_operand))
        pass

    if right_operand.requires_grad:
        """
        We need to define the gradient function of the right operand.
        And the right operand's grad shape is the same as the right operand's shape.
        """

        def grad_wrt_right_operand(grad: np.ndarray) -> np.ndarray:
            current_operand_grad = grad
            dimen_gap = grad.ndim - right_operand.data.ndim
            for _ in range(dimen_gap):
                current_operand_grad = current_operand_grad.sum(axis=0)
            for i, dimen in enumerate(right_operand.shape):
                if dimen == 1:
                    current_operand_grad = current_operand_grad.sum(
                        axis=i, keepdims=True
                    )
            return current_operand_grad

        child_nodes.append((right_operand, grad_wrt_right_operand))
        pass

    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="add",
    )
    return result_node


def _mul(left_operand: Carrot, right_operand: Carrot) -> Carrot:
    """
    mul: left_operand * right_operand
    """
    result_data = left_operand.data * right_operand.data
    requires_grad = left_operand.requires_grad or right_operand.requires_grad
    child_nodes = []
    if left_operand.requires_grad:
        """
        We need to define the gradient function of the left operand.
        And the left operand's grad shape is the same as the left operand's shape.
        """

        def grad_wrt_left_operand(grad: np.ndarray) -> np.ndarray:
            current_operand_grad = grad * right_operand.data
            dimen_gap = grad.ndim - left_operand.data.ndim
            for _ in range(dimen_gap):
                current_operand_grad = current_operand_grad.sum(axis=0)
            for i, dimen in enumerate(left_operand.shape):
                if dimen == 1:
                    current_operand_grad = current_operand_grad.sum(
                        axis=i, keepdims=True
                    )
            return current_operand_grad

        child_nodes.append((left_operand, grad_wrt_left_operand))
        pass

    if right_operand.requires_grad:
        """
        We need to define the gradient function of the right operand.
        And the right operand's grad shape is the same as the right operand's shape.
        """

        def grad_wrt_right_operand(grad: np.ndarray) -> np.ndarray:
            current_operand_grad = grad * left_operand.data
            dimen_gap = grad.ndim - right_operand.data.ndim
            for _ in range(dimen_gap):
                current_operand_grad = current_operand_grad.sum(axis=0)
            for i, dimen in enumerate(right_operand.shape):
                if dimen == 1:
                    current_operand_grad = current_operand_grad.sum(
                        axis=i, keepdims=True
                    )
            return current_operand_grad

        child_nodes.append((right_operand, grad_wrt_right_operand))
        pass

    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="mul",
    )
    return result_node


def _neg(operand: Carrot) -> Carrot:
    """
    neg: -operand
    """
    result_data = -operand.data
    requires_grad = operand.requires_grad
    child_nodes = []
    if operand.requires_grad:
        """
        We need to define the gradient function of the operand.
        And the operand's grad shape is the same as the operand's shape.
        """

        def grad_wrt_operand(grad: np.ndarray) -> np.ndarray:
            return -grad

        child_nodes.append((operand, grad_wrt_operand))
        pass

    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="neg",
    )
    return result_node


def _sub(left_operand: Carrot, right_operand: Carrot) -> Carrot:
    """
    sub: left_operand - right_operand
    """
    result_data = left_operand.data - right_operand.data
    requires_grad = left_operand.requires_grad or right_operand.requires_grad
    child_nodes = []
    if left_operand.requires_grad:
        """
        We need to define the gradient function of the left operand.
        And the left operand's grad shape is the same as the left operand's shape.
        """

        def grad_wrt_left_operand(grad: np.ndarray) -> np.ndarray:
            current_operand_grad = grad
            dimen_gap = grad.ndim - left_operand.data.ndim
            for _ in range(dimen_gap):
                current_operand_grad = current_operand_grad.sum(axis=0)
            for i, dimen in enumerate(left_operand.shape):
                if dimen == 1:
                    current_operand_grad = current_operand_grad.sum(
                        axis=i, keepdims=True
                    )
            return current_operand_grad

        child_nodes.append((left_operand, grad_wrt_left_operand))
        pass
    if right_operand.requires_grad:
        """
        We need to define the gradient function of the right operand.
        And the right operand's grad shape is the same as the right operand's shape.
        """

        def grad_wrt_right_operand(grad: np.ndarray) -> np.ndarray:
            current_operand_grad = -grad
            dimen_gap = grad.ndim - right_operand.data.ndim
            for _ in range(dimen_gap):
                current_operand_grad = current_operand_grad.sum(axis=0)
            for i, dimen in enumerate(right_operand.shape):
                if dimen == 1:
                    current_operand_grad = current_operand_grad.sum(
                        axis=i, keepdims=True
                    )
            return current_operand_grad

        child_nodes.append((right_operand, grad_wrt_right_operand))
        pass
    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="sub",
    )
    return result_node


def _pow(operand: Carrot, n: int) -> Carrot:
    """
    pow: operand ** n
    """
    result_data = operand.data**n
    requires_grad = operand.requires_grad
    child_nodes = []
    if operand.requires_grad:
        """
        We need to define the gradient function of the operand.
        And the operand's grad shape is the same as the operand's shape.
        """

        def grad_wrt_operand(grad: np.ndarray) -> np.ndarray:
            current_operand_grad = grad * n * operand.data ** (n - 1)
            return current_operand_grad

        child_nodes.append((operand, grad_wrt_operand))
        pass

    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="pow",
    )
    return result_node


def _getitem(operand: Carrot, slice_index) -> Carrot:
    """
    _getitem: carrot object[slice_index]
    """
    result_data = operand.data[slice_index]
    requires_grad = operand.requires_grad
    child_nodes = []
    if requires_grad:

        def grad_wrt_operand(grad: np.ndarray) -> np.ndarray:
            """
            we need to augument grad size, because slice operation decreases carrot size.
            """
            current_operand_grad = np.zeros_like(operand.data)
            current_operand_grad[slice_index] = grad
            return current_operand_grad

        child_nodes.append((operand, grad_wrt_operand))
        pass

    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="slice",
    )
    return result_node


def _reshape(operand: Carrot, reshaped_size) -> Carrot:
    """
    reshape carrot shape.
    """
    operand_shape = operand.data.shape
    result_data = operand.data.reshape(reshaped_size)
    requires_grad = operand.requires_grad
    child_nodes = []
    if requires_grad:

        def grad_wrt_operand(grad: np.ndarray) -> np.ndarray:
            """
            only need to reshape grad shape.
            """
            current_operand_grad = grad.reshape(operand_shape)
            return current_operand_grad

        child_nodes.append((operand, grad_wrt_operand))
        pass
    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="reshape",
    )
    return result_node


def _matmul(left_operand: Carrot, right_operand: Carrot) -> Carrot:
    """
    matmul: left_operand & right_operand.
    we need to use matrix diff, please read 《matrix cookbook》 for more details.
    """
    result_data = left_operand.data @ right_operand.data
    requires_grad = left_operand.requires_grad or right_operand.requires_grad
    child_nodes = []
    if left_operand.requires_grad:

        def grad_wrt_left_operand(grad: np.ndarray) -> np.ndarray:
            """
            matrx diff: C = AB, ∂C/∂A=grad @ B^T
            """
            return grad @ right_operand.data.T

        child_nodes.append((left_operand, grad_wrt_left_operand))
        pass

    if right_operand.requires_grad:

        def grad_wrt_right_operand(grad: np.ndarray) -> np.ndarray:
            """
            matrx diff: C = AB, ∂C/∂B= A^T @ grad
            """
            return left_operand.data.T @ grad

        child_nodes.append((right_operand, grad_wrt_right_operand))
        pass

    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="@",
    )
    return result_node


def carrot_sum(operand: Carrot) -> Carrot:
    """
    sum: operand.sum()
    """
    result_data = operand.data.sum()
    requires_grad = operand.requires_grad
    child_nodes = []
    if requires_grad:

        def grad_wrt_operand(grad: np.ndarray) -> np.ndarray:
            return np.ones_like(operand.data) * grad

        child_nodes.append((operand, grad_wrt_operand))
        pass
    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="sum",
    )
    return result_node


def carrot_mean(operand: Carrot) -> Carrot:
    """
    mean: operand.mean()
    """
    result_data = operand.data.mean()
    requires_grad = operand.requires_grad
    child_nodes = []
    if requires_grad:

        def grad_wrt_operand(grad: np.ndarray) -> np.ndarray:
            return grad * np.ones_like(operand.data) / len(operand.data)

        child_nodes.append((operand, grad_wrt_operand))
        pass
    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="mean",
    )
    return result_node


def _clip(operand: Carrot, min_value, max_value) -> Carrot:
    """
    let carrot data in a friendly range.
    not too min and not too max.
    """
    result_data = np.clip(operand.data, min_value, max_value)
    requires_grad = operand.requires_grad
    child_nodes = []
    if requires_grad:

        def grad_wrt_operand(grad: np.ndarray) -> np.ndarray:
            """
            let too min and too max value'grad be zero.
            """
            flat_grad = grad.copy().reshape(-1)
            flat_operand = operand.reshape(-1)
            mask = (flat_operand.data < min_value) | (flat_operand.data > max_value)
            flat_grad[mask] = 0
            return flat_grad.reshape(grad.shape)

        child_nodes.append((operand, grad_wrt_operand))
        pass
    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="clip",
    )
    return result_node


if __name__ == "__main__":
    pass