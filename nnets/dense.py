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

        grad = grad @ self.weights.T
        return grad


class DenseManualUpdate(Module):
    """
    Dense layer, but we manually update the weights and bias externally.
    """
    def __init__(self, input_size, output_size, dropout=None, activation=True, seed=0):
        self.add_activation = activation
        self.hidden = None
        self.prev_hidden = None

        # Initialize the weights.  They'll be in the range -sqrt(k) to sqrt(k), where k = 1 / input_size
        np.random.seed(seed)
        k = math.sqrt(1 / input_size)
        self.weights = np.random.rand(input_size, output_size) * (2 * k) - k

        # Our bias will be initialized to 1
        self.bias = np.ones((1,output_size))

    def forward(self, x):
        # Copy the layer input for backprop
        self.prev_hidden = x.copy()
        # Multiply the input by the weights, then add the bias
        x = np.matmul(x, self.weights) + self.bias
        # Apply the activation function
        if self.add_activation:
            x = np.maximum(x, 0)
        # Copy the layer output for backprop
        self.hidden = x.copy()
        return x

    def backward(self, grad):
        # "Undo" the activation function if it was added
        if self.add_activation:
            grad = np.multiply(grad, np.heaviside(self.hidden, 0))

        # Calculate the parameter gradients
        w_grad = self.prev_hidden.T @ grad # This is not averaged across the batch, due to the way matrix multiplication sums
        b_grad = np.mean(grad, axis=0) # This is averaged across the batch
        param_grads = [w_grad, b_grad]

        # Calculate the next layer gradient
        grad = grad @ self.weights.T
        return param_grads, grad

    def update(self, w_grad, b_grad):
        # Update the weights given an update matrix
        self.weights += w_grad
        self.bias += b_grad


def forward(x, layers, training=True):
    # Loop through each layer
    for layer in layers:
        # Run the forward pass
        layer.training = training
        x = layer.forward(x)
    return x


def backward(grad, layers):
    # Save the gradients for each layer
    layer_grads = []
    # Loop through each layer in reverse order (starting from the output layer)
    for layer in reversed(layers):
        # Get the parameter gradients and the next layer gradient
        param_grads, grad = layer.backward(grad)
        layer_grads.append(param_grads)
    return layer_grads


