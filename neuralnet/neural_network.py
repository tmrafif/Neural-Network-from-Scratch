import numpy as np
import pandas as pd

from .losses import mse, mse_prime

class NeuralNetwork:
    def __init__(self, layers):
        self.layers = layers

    def predict(self, input):
        output = input
        for layer in self.layers:
            output = layer.forward(output)
        return output
    
    def fit(self, X_train, y_train, epochs, learning_rate, loss=mse, loss_prime=mse_prime, verbose=True):
        error_hist = []
        accuracy_hist = []
        for epoch in range(epochs):
            error = 0
            for x, y in zip(X_train, y_train):
                # forward
                y_pred = self.predict(x)
                error += loss(y, y_pred)

                # backward
                model_grads = loss_prime(y, y_pred)
                for layer in reversed(self.layers):
                    model_grads = layer.backward(learning_rate, model_grads)

            error /= len(X_train)
            error_hist.append(error)
            
            if verbose:
                print(f"Epoch {epoch + 1}/{epochs}, error={error}")

        self.error_df = pd.DataFrame({
            'epoch': np.arange(1, epochs+1),
            'error': error_hist
            })
        