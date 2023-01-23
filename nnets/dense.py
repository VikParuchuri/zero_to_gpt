from network import Module
from activation import Relu
import numpy as np
import math

class Dense(Module):
    def __init__(self, input_size, output_size, bias=True, activation=True, seed=0):
        self.add_bias = bias
        self.add_activation = activation
        self.hidden = None
        self.prev_hidden = None

        np.random.seed(seed)
        k = math.sqrt(1 / input_size)
        self.weights = np.random.rand(input_size, output_size) * (2 * k) - k
        self.bias = np.ones((1, output_size)) * (2 * k) - k
        self.activation = Relu()

        super().__init__()

    def forward(self, x):
        self.prev_hidden = x.copy()
        x = np.matmul(x, self.weights)
        if self.add_bias:
            x += self.bias

        if self.add_activation:
            x = self.activation.forward(x)
        self.hidden = x.copy()
        return x

    def backward(self, grad, lr):
        if self.add_activation:
            grad = self.activation.backward(grad, lr, self.hidden)

        w_grad = self.prev_hidden.T @ grad
        b_grad = np.mean(grad, axis=0)

        self.weights -= w_grad * lr
        if self.add_bias:
            self.bias -= b_grad * lr

        grad = (self.weights @ grad.T).T
        return grad
