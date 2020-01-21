import numpy as np
import copy
from Layers.Base import Base
from Layers.Helpers import compute_bn_gradients


class BatchNormalization(Base):
    def __init__(self, channels):
        super().__init__()
        self.channels = channels
        # input shape : B x H (BMN X H for Conv)
        # weights, bias shape : 1 x H
        self.weights = np.ones((1, self.channels))
        self.bias = np.zeros((1, self.channels))
        # self.weights = []
        # self.bias = []
        self.mean = np.zeros((1, self.channels))
        self.var = np.zeros((1, self.channels))
        self.test_mean = None
        self.test_var = None
        self.decay = 0.8
        self.input_tensor = None
        self.input_shape = None
        self.norm = None
        # flag vector or image, vector + FullyConnected, image + CNN
        self.flag = None
        self._gradient_weights = None
        self._gradient_bias = None
        # for gradient update
        self._optimizer = None
        self.weights_optimizer = None
        self.bias_optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        self.weights = np.ones((1, self.channels))
        self.bias = np.zeros((1, self.channels))

    def forward(self, input_tensor):
        self.input_shape = input_tensor.shape

        # reformat image to vector
        if len(input_tensor.shape) == 4:
            self.input_tensor = self.reformat(input_tensor)
            self.flag = "image"
        else:
            self.input_tensor = input_tensor
            self.flag = "vector"

        self.mean = np.mean(self.input_tensor, axis=0)
        self.var = np.var(self.input_tensor, axis=0)

        # for training
        if self.phase == "train":
            input_tensor_norm = np.divide((self.input_tensor - self.mean), np.sqrt(self.var + np.finfo(float).eps))
            # for test (moving average estimation of training set mean and variance)
            if self.test_mean is None and self.test_var is None:
                self.test_mean = self.mean
                self.test_var = self.var
            self.test_mean = self.decay * self.test_mean + (1 - self.decay) * self.mean
            self.test_var = self.decay * self.test_var + (1 - self.decay) * self.var
        else:
            input_tensor_norm = np.divide((self.input_tensor - self.test_mean), np.sqrt(self.test_var + np.finfo(float).eps))

        self.norm = input_tensor_norm
        output_tensor = input_tensor_norm * self.weights + self.bias

        # for image, reformat in from (BMN, H) back to (B, H, M, N)
        if self.flag == "image":
            output_tensor = self.reformat(output_tensor)
        return output_tensor

    def backward(self, error_tensor):
        # reformat image to vector
        if len(error_tensor.shape) == 4:
            error_tensor = self.reformat(error_tensor)

        # compute gradient of gamma and beta
        self._gradient_weights = np.sum(error_tensor * self.norm, axis=0).reshape(1, -1)
        self._gradient_bias = np.sum(error_tensor, axis=0).reshape(1, -1)

        # update gradient
        if self.weights_optimizer is not None:
            self.weights = self.weights_optimizer.calculate_update(self.weights, self._gradient_weights)
        if self.bias_optimizer is not None:
            self.bias = self.bias_optimizer.calculate_update(self.bias, self._gradient_bias)

        # computation of the gradient w.r.t.inputs
        output_tensor = compute_bn_gradients(error_tensor, self.input_tensor, self.weights, self.mean, self.var)

        if self.flag == "image":
            output_tensor = self.reformat(output_tensor)
        return output_tensor

    def reformat(self, tensor):
        # from image tensor to vector tensor
        B, H, M, N = self.input_shape
        if len(tensor.shape) == 4:
            # (B, H, M, N) -> (B, H, MN) -> (B, MN, H) -> (BMN, H)
            tensor = tensor.reshape(B, H, M*N)
            tensor = np.transpose(tensor, (0, 2, 1))
            tensor = tensor.reshape(B * M * N, H)
        else:
            # (BMN, H) -> (B, MN, H) -> (B, H, MN) -> (B, H, M, N)
            tensor = tensor.reshape(B, M * N, H)
            tensor = np.transpose(tensor, (0, 2, 1))
            tensor = tensor.reshape((B, H, M, N))
        return tensor

    # two properties
    def get_gradient_weights(self):
        return self._gradient_weights

    def set_gradient_weights(self, gradient_weights):
        self._gradient_weights = gradient_weights

    gradient_weights = property(get_gradient_weights, set_gradient_weights)

    def get_gradient_bias(self):
        return self._gradient_bias

    def set_gradient_bias(self, gradient_bias):
        self._gradient_bias = gradient_bias

    gradient_bias = property(get_gradient_bias, set_gradient_bias)

    # property optimizer
    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer
        self.weights_optimizer = copy.deepcopy(self._optimizer)
        self.bias_optimizer = copy.deepcopy(self._optimizer)

    optimizer = property(get_optimizer, set_optimizer)

    # calculate sum of weights in each layer for regularization
    @property
    def regularizer_loss(self):
        delete_bias_weight = np.delete(self.weights, -1, 1)
        sum_weight = 0
        if self._optimizer is not None:
            if self._optimizer.regularizer is not None:
                sum_weight = self._optimizer.regularizer.norm(delete_bias_weight)
        return sum_weight
