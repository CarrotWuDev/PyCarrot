from ..carrot import Carrot
import numpy as np


# activation functions
def sigmoid(operand: Carrot) -> Carrot:
    """
    sigmoid: activation.
    """
    result_data = 1 / (1 + np.exp(operand.data))
    requires_grad = operand.requires_grad
    child_nodes = []
    if requires_grad:

        def grad_wrt_operand(grad: np.ndarray) -> np.ndarray:
            return grad * (result_data * (1 - result_data))

        child_nodes.append((operand, grad_wrt_operand))
        pass

    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="sigmoid",
    )
    return result_node


def softmax(operand: Carrot) -> Carrot:
    """
    softmax: activation.
    """
    exp_data = np.exp(operand.data - np.max(operand.data, axis=-1, keepdims=True))
    result_data = exp_data / np.sum(exp_data, axis=-1, keepdims=True)
    requires_grad = operand.requires_grad
    child_nodes = []
    if requires_grad:
        def grad_wrt_operand(grad: np.ndarray) -> np.ndarray:
            s = result_data.reshape(-1, 1)
            current_operand_grad = (s * (np.eye(s.shape[0]) - s.T)) * grad
            # dimen_gap = current_operand_grad.ndim - operand.data.ndim
            # for _ in range(dimen_gap):
            #     current_operand_grad = current_operand_grad.sum(axis=0)
            return current_operand_grad

        child_nodes.append((operand, grad_wrt_operand))

    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="softmax",
    )
    return result_node


def tanh(operand: Carrot) -> Carrot:
    """
    sigmoid: activation.
    """
    result_data = np.tanh(operand.data)
    requires_grad = operand.requires_grad
    child_nodes = []
    if requires_grad:

        def grad_wrt_operand(grad: np.ndarray) -> np.ndarray:
            return grad * (1 - result_data**2)

        child_nodes.append((operand, grad_wrt_operand))
        pass

    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="tanh",
    )
    return result_node


def relu(operand: Carrot) -> Carrot:
    result_data = np.clip(operand.data, 0, np.inf)
    requires_grad = operand.requires_grad
    child_nodes = []
    if requires_grad:

        def grad_wrt_operand(grad: np.ndarray) -> np.ndarray:
            return grad * (operand.data > 0).astype(np.float64)

        child_nodes.append((operand, grad_wrt_operand))
        pass

    result_node = Carrot(
        data=result_data,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="relu",
    )
    return result_node


# loss funcions
def mse_loss(predicted: Carrot, target: Carrot) -> Carrot:
    """
    mse: mean square error
    """
    # type 1: return expression, but more nodes and low speed
    # return ((predicted - target) ** 2).mean()

    # type 2: consider grad
    number = len(predicted)
    loss = np.mean((predicted.data - target.data) ** 2)
    requires_grad = predicted.requires_grad or target.requires_grad
    child_nodes = []
    if predicted.requires_grad:

        def grad_wrt_predicted(grad):
            return 2 * (predicted.data - target.data) * grad / number

        child_nodes.append((predicted, grad_wrt_predicted))
        pass

    if target.requires_grad:

        def grad_wrt_target(grad):
            return 2 * (target.data - predicted.data) * grad / number

        child_nodes.append((target, grad_wrt_target))
        pass

    result_node = Carrot(
        data=loss,
        requires_grad=requires_grad,
        child_nodes=child_nodes,
        name="mse",
    )
    return result_node


def cross_entropy_loss(predicted: Carrot, target: Carrot) -> Carrot:
    pass


if __name__ == "__main__":
    # test sigmoid
    # x = np.array([1, 2])
    # x = Carrot(x, requires_grad=True, name="x")
    # y = sigmoid(x)
    # print(x.data)
    # print("---------------")
    # print(y.data)
    # y.backward()
    # print("---------------")
    # print(y.grad.data)
    # print("---------------")
    # print(x.grad.data)

    # test relu
    # x = np.array([[-1, 2, 3], [1, -1, 3], [1, 2, -1]])
    # x = Carrot(x, requires_grad=True, name="x")
    # y = relu(x)
    # print(x.data)
    # print("---------------")
    # print(y.data)
    # y.backward()
    # print("---------------")
    # print(y.grad.data)
    # print("---------------")
    # print(x.grad.data)

    # test softmax
    x = np.array([1, 2, 3, 4])
    x = Carrot(x, requires_grad=True, name="x")
    y = softmax(x)
    print(x.data)
    print("---------------")
    print(y.data)
    y.backward()
    print("---------------")
    print(y.grad.data)
    print("---------------")
    print(x.grad.data)

    # test mse_loss
    # target = np.array([[1], [2], [3]])
    # target = Carrot(target, requires_grad=True)
    # pred = np.array([[1.5], [2.5], [3.5]])
    # pred = Carrot(pred, requires_grad=True)
    # loss = mse_loss(predicted=pred, target=target)
    # print(target.data)
    # print("--------------------")
    # print(pred.data)
    # print("--------------------")
    # print(loss.data, loss.name)
    # loss.backward()
    # print("--------------------")
    # print(loss.grad.data)
    # print("--------------------")
    # print(target.grad.data)
    # print("--------------------")
    # print(pred.grad.data)
