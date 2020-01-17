import numpy as np

class ReLU:

    def __init__(self):
        self.input_tensor = None

    def forward(self, input_tensor):
        self.input_tensor = input_tensor
        # input_tensor = np.maximum(0, input_tensor)
        input_tensor = np.where(input_tensor < 0, 0, input_tensor)
        return input_tensor

    def backward(self, error_tensor):
        # relu_gradient = self.input_tensor > 0
        # error_tensor = error_tensor * relu_gradient
        error_tensor = np.where(self.input_tensor < 0, 0, error_tensor)
        return error_tensor