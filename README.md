# jacob

A toy machine learning and automatic differentiation library which prioritizes mathematical consistency, implementation simplicity, and readability over performance or convenience.

### Important Notes

-   `jacob` intentionally shares design conventions with PyTorch.
-   `jacob` uses the `numpy` library for numerical operations. No other dependencies are required.
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
        h = self.W1 @ x + self.b1
        h = relu(h)
        y = self.W2 @ h + self.b2
        return sigmoid(y)


N = 100

net = MyNeuralNet()
x = Tensor(np.random.randn(N, 2, 1))
target = Tensor(np.random.randn(N, 1, 1))


EPOCHS = 100
LR = 0.01

for _ in range(EPOCHS):
    # Notice that the gradient dloss/dW1 is a Nx3x2 tensor, where each element
    # is the gradient of the loss with respect to the corresponding training example

    y = net.forward(x)

    loss = ((y - target) ** 2).sum()
    loss.backward()

    print(loss.np.mean())

    # set_value() automatically zeros the gradients
    net.W1.set_value(net.W1.np - LR * net.W1.grad)
    net.b1.set_value(net.b1.np - LR * net.b1.grad)
    net.W2.set_value(net.W2.np - LR * net.W2.grad)
    net.b2.set_value(net.b2.np - LR * net.b2.grad)

```
