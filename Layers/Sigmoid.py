from Layers.Base import Base
import numpy as np

# sigmoid(x) = 1/((e**(-x))+1)
# forward: output is sigmoid(input_tensor)
# backward: output is error_tensor * (derivative of sigmoid(x))
# derivative of sigmoid(x) = f(x) * (1 − f(x))


class Sigmoid(Base):
    def __init__(self):
        super().__init__()
        self.activations = None

    def forward(self, input_tensor):
        self.activations = 1 / (np.exp(-1 * input_tensor) + 1)
        return self.activations

    def backward(self, error_tensor):
        sigmoid_prime = self.activations * (1 - self.activations)
        error_tensor = error_tensor * sigmoid_prime
        return error_tensor

