from network import Module
import numpy as np

class Relu(Module):
    def __init__(self):
        self.relu = lambda x: np.maximum(x, 0)
        self.hidden = None

    def forward(self, x):
        self.hidden = x.copy()
        return self.relu(x)

    def backward(self, grad, lr, prev_hidden=None):
        return np.multiply(grad, np.heaviside(self.hidden, 0))
