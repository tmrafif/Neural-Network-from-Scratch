import numpy as np
from scipy import signal

from .layer import Layer

class Convolutional(Layer):
    def __init__(self, input_shape: tuple, kernel_size: int, output_depth: int):
        input_depth, input_height, input_width = input_shape
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_depth = output_depth

        self.kernels_shape = (output_depth, input_depth, kernel_size, kernel_size)
        self.kernels = np.random.randn(*self.kernels_shape)

        self.output_shape = (output_depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.biases = np.random.randn(*self.output_shape)

    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        
        for i in range(self.output_depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d(self.input[j], self.kernels[i, j], "valid")

        return self.output
    
    def backward(self, learning_rate, output_gradient):
        kernels_gradient = np.zeros(self.kernels_shape)
        inputs_gradient = np.zeros(self.input_shape)

        for i in range(self.output_depth):
            for j in range(self.input_depth):
                kernels_gradient[i, j] = signal.correlate2d(self.input[j], output_gradient[i], "valid")
                inputs_gradient[j] += signal.convolve2d(output_gradient[i], self.kernels[i, j], "full")

        self.kernels -= learning_rate * kernels_gradient
        self.biases -= learning_rate * output_gradient
        return inputs_gradient