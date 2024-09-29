from typing import Iterable
from .core import DifferentiableF, Matmul, Tensor
import numpy as np


def matmul(a: Tensor, b: Tensor) -> Tensor:
    return Matmul()(a, b)


def norm(x: Tensor) -> Tensor:
    return Norm()(x)


class Norm(DifferentiableF):
    def forward(self, input: np.ndarray) -> np.ndarray:
        return np.array(np.linalg.norm(input))

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        return (grad_wrt_output * self.fwd_inputs[0].np / self.fwd_value,)


class Sum(DifferentiableF):
    def forward(self, *inputs: np.ndarray) -> np.ndarray:
        a = inputs[0]

        for b in inputs[1:]:
            a += b

        return a

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        return [grad_wrt_output] * len(self.fwd_inputs)


def sum(*inputs: Tensor) -> Tensor:
    return Sum()(*inputs)


class Sigmoid(DifferentiableF):
    def forward(self, input: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-input))

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        sigmoid = self.fwd_value
        return (grad_wrt_output * sigmoid * (1 - sigmoid),)


def sigmoid(x: Tensor) -> Tensor:
    return Sigmoid()(x)


class Tanh(DifferentiableF):
    def forward(self, input: np.ndarray) -> np.ndarray:
        return np.tanh(input)

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        return (grad_wrt_output * (1 - self.fwd_value**2),)


def tanh(x: Tensor) -> Tensor:
    return Tanh()(x)


class Relu(DifferentiableF):
    def forward(self, input: np.ndarray) -> np.ndarray:
        return np.maximum(0, input)

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        return (grad_wrt_output * (self.fwd_value > 0),)


def relu(x: Tensor) -> Tensor:
    return Relu()(x)


class Softmax(DifferentiableF):
    def forward(self, input: np.ndarray) -> np.ndarray:
        exps = np.exp(input - np.max(input))
        return exps / np.sum(exps)

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        softmax = self.fwd_value
        return (grad_wrt_output * softmax * (1 - softmax),)


def softmax(x: Tensor) -> Tensor:
    return Softmax()(x)


class NormSquared(DifferentiableF):
    def forward(self, input: np.ndarray) -> np.ndarray:
        return np.linalg.norm(input) ** 2  # type: ignore

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        return (grad_wrt_output * 2 * self.fwd_inputs[0].np,)


def norm_squared(x: Tensor) -> Tensor:
    return NormSquared()(x)


class Exp(DifferentiableF):
    def forward(self, input: np.ndarray) -> np.ndarray:
        return np.exp(input)

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        return (grad_wrt_output * np.exp(self.fwd_value),)


def exp(x: Tensor) -> Tensor:
    return Exp()(x)
