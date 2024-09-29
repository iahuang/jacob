from __future__ import annotations
from typing import Iterable
import numpy as np


class DifferentiableF:
    fwd_inputs: tuple[Tensor, ...]
    fwd_value: np.ndarray

    def __call__(self, *inputs: Tensor) -> Tensor:
        self.fwd_inputs = inputs
        val = Tensor(self.forward(*[i._value for i in inputs]))
        val._derived_from = self
        self.fwd_value = val._value

        return val

    def forward(self, *args, **kwargs) -> np.ndarray:
        """
        Given the input values, compute the output of this function. This function should be overridden by subclasses.
        This method should also not be invoked directly, but rather through the `__call__` method.
        """

        raise NotImplementedError

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        """
        Given the gradient of some variable X with respect to the output of this function, compute the gradient of X
        with respect to the input of this function. More specifically, if this function is `f(x1, x2, ..., xn)`, then given
        the value of `dX/df` at `x1, x2, ..., xn`, this function should return the values of `dX/dx1, dX/dx2, ..., dX/dxn`.
        """

        raise NotImplementedError


class Tensor:
    _value: np.ndarray
    _derived_from: DifferentiableF | None
    _transposition_of: Tensor | None
    grad: np.ndarray

    def __init__(self, value: np.ndarray | float | list) -> None:
        if isinstance(value, (float, int)):
            value = np.array(value)

        if isinstance(value, list):
            value = np.array(value)

        shape = value.shape

        # transform scalars into 1D array

        if len(shape) == 0:
            value = value.reshape((1, 1))
        elif len(shape) == 1:
            value = value.reshape((shape[0], 1))

        self._value = value
        self._derived_from = None
        self.grad = np.zeros_like(value)
        self._transposition_of = None

    def backward(self) -> None:
        self._backward(np.ones_like(self._value))

    def _backward(self, grad: np.ndarray) -> None:
        self.grad = grad

        if self._transposition_of is not None:
            self._transposition_of._backward(grad.T)
            return

        if self._derived_from is None:
            return

        f = self._derived_from
        f_inputs = self._derived_from.fwd_inputs
        for i, grad_wrt_input in enumerate(f.backward(self.grad)):
            # make sure that grad_wrt_input has the same shape as the input
            if grad_wrt_input.shape != f_inputs[i].shape:
                print(
                    f"Warning: input gradient from {f.__class__.__name__} expected to have shape {f_inputs[i].shape}, but has shape {grad_wrt_input.shape}"
                )
            self._derived_from.fwd_inputs[i]._backward(grad_wrt_input)

    @property
    def shape(self) -> tuple[int, ...]:
        return self._value.shape

    @property
    def T(self) -> Tensor:
        val = Tensor(self._value.T)
        val._transposition_of = self

        return val

    @property
    def np(self) -> np.ndarray:
        return self._value

    def __add__(self, other: Tensor) -> Tensor:
        return Add()(self, other)

    def __radd__(self, other: Tensor) -> Tensor:
        return Add()(other, self)

    def __sub__(self, other: Tensor) -> Tensor:
        return Subtract()(self, other)

    def __rsub__(self, other: Tensor) -> Tensor:
        return Subtract()(other, self)

    def __matmul__(self, other: Tensor) -> Tensor:
        return Matmul()(self, other)

    def __rmatmul__(self, other: Tensor) -> Tensor:
        return Matmul()(other, self)

    def __mul__(self, other: Tensor | float) -> Tensor:
        if isinstance(other, (float, int)):
            return ScalarMultiply(other)(self)

        return ElementMultiply()(self, other)

    def __rmul__(self, other: Tensor | float) -> Tensor:
        if isinstance(other, (float, int)):
            return ScalarMultiply(other)(self)

        return ElementMultiply()(other, self)

    def __neg__(self) -> Tensor:
        return ScalarMultiply(-1)(self)

    def __pow__(self, power: float) -> Tensor:
        return ElementPower(power)(self)

    def sum(self) -> Tensor:
        return SelfSum()(self)

    @classmethod
    def zero(cls) -> Tensor:
        return cls(np.zeros((1, 1)))

    @classmethod
    def zero_vector(cls, n: int) -> Tensor:
        return cls(np.zeros((n,)))

    @classmethod
    def zeros(cls, shape: tuple[int, ...]) -> Tensor:
        return cls(np.zeros(shape))

    def zero_grad(self) -> None:
        self.grad = np.zeros_like(self._value)

    def set_value(self, value: np.ndarray) -> None:
        self._value = value
        self.zero_grad()
        self._derived_from = None


class ScalarMultiply(DifferentiableF):
    coeff: float

    def __init__(self, coeff: float) -> None:
        self.coeff = coeff

    def forward(self, input: np.ndarray) -> np.ndarray:
        return self.coeff * input

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        return (grad_wrt_output * self.coeff,)


class Add(DifferentiableF):
    def forward(self, input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
        return input1 + input2

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        return grad_wrt_output, grad_wrt_output


class Subtract(DifferentiableF):
    def forward(self, input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
        return input1 - input2

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        return grad_wrt_output, -grad_wrt_output


class ElementMultiply(DifferentiableF):
    def forward(self, input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
        return input1 * input2

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        return (
            grad_wrt_output * self.fwd_inputs[1].np,
            grad_wrt_output * self.fwd_inputs[0].np,
        )


class Matmul(DifferentiableF):
    def forward(self, input1: np.ndarray, input2: np.ndarray) -> np.ndarray:
        return input1 @ input2

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        transposed_fwd_inputb = self.fwd_inputs[1].np.T
        transposed_fwd_inputa = self.fwd_inputs[0].np.T

        if len(grad_wrt_output.shape) == 3:
            transposed_fwd_inputb = self.fwd_inputs[1].np.transpose((0, 2, 1))

        if len(self.fwd_inputs[0].shape) == 3:
            transposed_fwd_inputa = self.fwd_inputs[0].np.transpose((0, 2, 1))

        return (
            grad_wrt_output @ transposed_fwd_inputb,
            transposed_fwd_inputa @ grad_wrt_output,
        )


class SelfSum(DifferentiableF):
    def forward(self, input: np.ndarray) -> np.ndarray:
        if len(input.shape) == 3:
            return np.sum(input, axis=1, keepdims=True)

        return np.sum(input)

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        return (grad_wrt_output * np.ones_like(self.fwd_inputs[0].np),)


class ElementPower(DifferentiableF):
    power: float

    def __init__(self, power: float) -> None:
        self.power = power

    def forward(self, input: np.ndarray) -> np.ndarray:
        return input**self.power

    def backward(self, grad_wrt_output: np.ndarray) -> Iterable[np.ndarray]:
        return (grad_wrt_output * self.power * self.fwd_value ** (self.power - 1),)
