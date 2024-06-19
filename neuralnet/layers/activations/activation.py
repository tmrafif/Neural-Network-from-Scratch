import numpy as np

from ..layer import Layer

class Activation(Layer):
    def __init__(self, func, func_prime):
        self.func = func
        self.func_prime = func_prime

    def forward(self, input):
        self.input = input
        self.output = self.func(self.input)
        return self.output

    def backward(self, learning_rate, output_gradient):
        return np.multiply(output_gradient, self.func_prime(self.input))