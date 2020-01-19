from Layers.Base import Base
import numpy as np

# forward: output = tanh(x) = ((e**x)-(e**(-x)))/((e**x)+(e**(-x))), x is input_tensor
# backward: output = error_tensor * (derivative of tanh(x))
# derivative of tanh(x) = 1 - tanh(x)^2


class TanH(Base):
    def __init__(self):
        super().__init__()
        # save activation function instead of input
        self.activations = None

    def forward(self, input_tensor):
        neg_input = (-1) * input_tensor
        self.activations = np.divide(np.exp(input_tensor) - np.exp(neg_input), np.exp(input_tensor) + np.exp(neg_input))
        return self.activations

    def backward(self, error_tensor):
        tanh_prime = 1 - np.power(self.activations, 2)
        error_tensor = error_tensor * tanh_prime
        return error_tensor
