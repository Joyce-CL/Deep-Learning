import numpy as np

class SoftMax:

    def __init__(self):
        self.estimated_class_pro = None

    def forward(self, input_tensor):
        # if input_tensor is so big, eg:10000, exp can not be calculated, so first do normalization on input_tensor
        input_tensor = input_tensor - input_tensor.max()
        # calculate exp y^_k = exp(x_k) / sum(exp(x))
        input_tensor_exp = np.exp(input_tensor)
        input_tensor_sum = np.sum(np.exp(input_tensor), axis=1, keepdims=True)
        self.estimated_class_pro = np.divide(input_tensor_exp, input_tensor_sum)
        return np.copy(self.estimated_class_pro)

    def backward(self, error_tensor):
        # E = estimated_class_pro(error_tensor - sum(estimated_class_pro * error_tensor))
        tmp = np.sum(error_tensor * self.estimated_class_pro, axis=1, keepdims=True)
        error_tensor = self.estimated_class_pro * (error_tensor - tmp)
        return error_tensor
