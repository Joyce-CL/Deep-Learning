import numpy as np

class Flatten:
    def __init__(self):
        self.input_tensor_shape = None

    def forward(self, input_tensor):
        # reshape input_tensor from eg:(9, 3, 4, 11) to (9, 3*4*11) for FNN forward input
        self.input_tensor_shape = input_tensor.shape
        input_tensor = input_tensor.reshape(input_tensor.shape[0], -1)
        return input_tensor

    def backward(self, error_tensor):
        # reshape error_tensor from eg:(9, 3*4*11) to (9, 3, 4, 11) for CNN backward input
        error_tensor = error_tensor.reshape(self.input_tensor_shape)
        return error_tensor
