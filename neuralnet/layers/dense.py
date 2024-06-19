import numpy as np

from .layer import Layer

class Dense(Layer):
    def __init__(self, input_size: int, output_size: int):
        self.weights = np.random.randn(output_size, input_size)
        self.biases = np.random.randn(output_size, 1)

    def forward(self, input):
        self.input = input
        self.output = np.dot(self.weights, input) + self.biases

        return self.output

    def backward(self, learning_rate, output_gradient):
        # self.biases_gradient = np.sum(output_gradient, axis=0, keepdims=True)
        weights_gradient = np.dot(output_gradient, self.input.T)
        inputs_gradient = np.dot(self.weights.T, output_gradient)
        
        # update weights and biases
        self.weights -= learning_rate * weights_gradient
        self.biases -= learning_rate * output_gradient
        
        return inputs_gradient