import numpy as np

from .activation import Activation
from ..layer import Layer

class Tanh(Activation):
    def __init__(self):
        def tanh(x):
            return np.tanh(x)
        
        def tanh_prime(x):
            return 1 - (np.tanh(x) ** 2)
        
        super().__init__(tanh, tanh_prime)

class Sigmoid(Activation):
    def __init__(self):
        def sigmoid(x):
            return 1 / (1 + np.exp(-x))
        
        def sigmoid_prime(x):
            s = sigmoid(x)
            return s * (1 - s)
        
        super().__init__(sigmoid, sigmoid_prime)

class Softmax(Layer):
    def forward(self, input):
        tmp = np.exp(input)
        self.output = tmp / np.sum(tmp, axis=1, keepdims=True)
        return self.output
    
    def backward(self, learning_rate, output_gradient):
        n = np.size(self.output)
        inputs_gradient = np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)
        return inputs_gradient