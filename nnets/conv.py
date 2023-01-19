from network import Module
from activation import Relu
import numpy as np
from skimage.util import view_as_windows
import math

def convolve(image, kernel):
    return np.matmul(image, kernel.reshape(math.prod(kernel.shape), 1)).copy()

def unroll_image(image, kernel_x, kernel_y):
    unrolled = view_as_windows(image, (kernel_x, kernel_y))
    x_size = (image.shape[0] - (kernel_x - 1))
    y_size = (image.shape[1] - (kernel_y - 1))
    rows = x_size * y_size
    return unrolled.reshape(rows, kernel_x * kernel_y)

class Conv(Module):
    def __init__(self, input_channels, output_channels, kernel_x, kernel_y, bias=True, activation=True, seed=0):
        self.add_bias = bias
        self.add_activation = activation
        self.kernel_x = kernel_x
        self.kernel_y = kernel_y
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden = None

        np.random.seed(seed)
        k = math.sqrt(1 / (input_channels * (kernel_x + kernel_y)))
        self.weights = np.random.rand(input_channels, output_channels, kernel_x, kernel_y) * (2 * k) - k
        self.bias = np.ones(output_channels) * (2 * k) - k
        self.activation = Relu()

        super().__init__()

    def forward(self, x):
        self.prev_hidden = x.copy()
        new_x = x.shape[1] - (self.kernel_x - 1)
        new_y = x.shape[2] - (self.kernel_y - 1)
        output = np.zeros((self.output_channels, new_x, new_y))
        for channel in range(self.input_channels):
            unrolled = unroll_image(x[channel, :], self.kernel_x, self.kernel_y)
            for next_channel in range(self.output_channels):
                kernel = self.weights[channel, next_channel, :]
                mult = convolve(unrolled, kernel).reshape(new_x, new_y)
                output[next_channel, :] += mult
        output /= x.shape[0]

        if self.add_bias:
            for next_channel in range(self.output_channels):
                output[next_channel, :] += self.bias[next_channel]


        if self.add_activation:
            output = self.activation.forward(output)
        self.hidden = output.copy()
        return output

    def backward(self, grad, lr):
        grad = grad.reshape(self.hidden.shape)
        if self.add_activation:
            grad = self.activation.backward(grad, lr, self.hidden)

        _, grad_x, grad_y = grad.shape
        new_grad = np.zeros(self.prev_hidden.shape)
        # Kernel weight update
        for channel in range(self.input_channels):
            # With multi-channel output, you need to loop across the output grads to link to input channel kernels
            # Each kernel gets a unique update
            flat_input = unroll_image(self.prev_hidden[channel, :], grad_x, grad_y)
            for next_channel in range(self.output_channels):
                # Kernel update
                channel_grad = grad[next_channel, :]
                # Each parameter is linked to multiple output pixels.
                # Dividing by the number of pixels ensures correct update size.
                grad_norm = math.prod(channel_grad.shape)
                k_grad = convolve(flat_input, channel_grad).reshape(self.kernel_x, self.kernel_y) / grad_norm
                self.weights[channel, next_channel, :] -= k_grad * lr
        # Bias update
        if self.add_bias:
            for next_channel in range(self.output_channels):
                channel_grad = grad[next_channel, :]
                self.bias[next_channel] -= np.mean(channel_grad) * lr

        # Propagate grad to next layer
        for next_channel in range(self.output_channels):
            channel_grad = grad[next_channel, :]
            padded_grad = np.pad(channel_grad, ((self.kernel_x - 1, self.kernel_x - 1), (self.kernel_y - 1, self.kernel_y - 1)))
            flat_padded = unroll_image(padded_grad, self.kernel_x, self.kernel_y)
            for channel in range(self.input_channels):
                # Grad to lower layer
                flipped_kernel = np.flip(self.weights[channel, next_channel, :], axis=[0, 1])
                updated_grad = convolve(flat_padded, flipped_kernel).reshape(self.prev_hidden.shape[1], self.prev_hidden.shape[2])
                # Since we're multiplying each input by multiple kernel values, reduce the gradient accordingly
                # This will reduce the edges more than necessary (they contribute to fewer output values), but is simple to implement
                new_grad[channel, :] += updated_grad / math.prod(flipped_kernel.shape)
        return new_grad