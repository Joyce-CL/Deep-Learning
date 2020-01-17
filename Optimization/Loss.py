import numpy as np

class CrossEntropyLoss:

    def __init__(self):
        self.input_tensor = None
        # self.softmax = None

    def forward(self, input_tensor, label_tensor):
        # softmax = np.exp(input_tensor)
        # softmax = softmax / np.sum(softmax, axis=1, keepdims=True)
        # self.softmax = softmax
        # Loss = sum(1-b)(-ln(y^_K + eps), when y_k = 1, here y^_k is input_tensor, y_k is label_tensor)
        self.input_tensor = input_tensor
        tmp = - np.log(input_tensor + np.finfo(float).eps)
        entropy_loss = np.sum(label_tensor * tmp)
        return entropy_loss

    def backward(self, label_tensor):
        error_tensor = - np.divide(label_tensor, self.input_tensor)
        return error_tensor
