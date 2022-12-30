from network import Module
import numpy as np

class Dense(Module):
    def __init__(self, input_size, output_size, bias=True, relu=True):
        self.add_bias = bias
        self.add_relu = relu
        self.hidden = None
        self.input = None

        self.weights = np.random.rand(input_size, output_size) / 5 - .1
        self.bias = np.ones((1, output_size))
        self.relu = lambda x: np.maximum(x, 0)

        super().__init__()

    def forward(self, x):
        self.input = x.copy()
        x = np.matmul(x, self.weights)
        if self.add_bias:
            x += self.bias
        self.hidden = x.copy()

        if self.add_relu:
            x = self.relu(x)
        return x

    def backward(self, grad, lr):
        if self.add_relu:
            grad = np.multiply(grad, np.heaviside(self.hidden, 1))

        grad = grad.T
        w_grad = np.matmul(grad, self.input).T
        b_grad = grad.T

        self.weights -= (w_grad + self.weights * .01) * lr
        if self.add_bias:
            self.bias -= b_grad * lr

        grad = np.matmul(self.weights, grad).T
        return grad
