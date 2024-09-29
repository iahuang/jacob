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
y = net.forward(x)

loss = ((y - target) ** 2).sum()
loss.backward()

# Notice that the gradient dloss/dW1 is a Nx3x2 tensor, where each element
# is the gradient of the loss with respect to the corresponding training example
print("W1 grad:", net.W1.grad.sum(axis=0) / N)
print("b1 grad:", net.b1.grad.sum(axis=0) / N)