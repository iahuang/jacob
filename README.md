# jacob

A toy machine learning and automatic differentiation library which prioritizes mathematical consistency, implementation simplicity, and readability over performance or convenience.

### Important Notes

-   `jacob` uses the `numpy` library for numerical operations.
-   `jacob.Tensor` objects are used to represent all numerical data. Canonically, the `Tensor` object
    only support two shapes: `(N, d1, d2)` and `(d1, d2)`. The former is used to represent a batch
    of `N` matrices of shape `(d1, d2)`, while the latter is used to represent a single matrix of
    shape `(d1, d2)`.
-   If a `Tensor` object is initialized with a scalar value, it is interpreted as a matrix of shape `(1, 1)`.
-   If a `Tensor` object is initialized with a 1D array, it is interpreted as a single-column matrix of shape `(d, 1)`.

### Usage

```python
import numpy as np
from jacob import Tensor, matmul, relu, sum
from jacob.funcs import sigmoid


class MyNeuralNet:
    def __init__(self):
        self.W1 = Tensor(np.random.randn(3, 2))
        self.b1 = Tensor(np.random.randn(3))
        self.W2 = Tensor(np.random.randn(1, 3))
        self.b2 = Tensor(np.random.randn(1))

    def forward(self, x: Tensor) -> Tensor:
        h = self.W1,@ x + self.b1
        h = relu(h)
        y = self.W2 @ h + self.b2
        return sigmoid(y)


N = 100

net = MyNeuralNet()
x = Tensor(np.random.randn(N, 2, 1))
target = Tensor(np.random.randn(N, 1, 1))
y = net.forward(x)

loss = ((y - target) ** 2).sum()
loss.backward()

# Notice that the gradient dloss/dW1 is a Nx3x2 tensor, where each element
# is the gradient of the loss with respect to the corresponding training example
print("W1 grad:", net.W1.grad.sum(axis=0) / N)
print("b1 grad:", net.b1.grad.sum(axis=0) / N)
```
