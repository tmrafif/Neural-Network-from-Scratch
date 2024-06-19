import numpy as np

from .layer import Layer

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        self.input = input
        return np.reshape(self.input, self.output_shape)

    def backward(self, learning_rate, output_gradient):
        return np.reshape(output_gradient, self.input_shape)
