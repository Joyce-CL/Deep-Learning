import numpy as np
import math
import scipy.signal as sgl
import copy

class Conv:

    def __init__(self, stride_shape, convolution_shape, num_kernels):
        # if stride_shape is a value, it means the kernel shifts the same step in horizontal & vertical direction
        # generalization stride_shape in form (a, b)
        # a: represent the stride in horizontal direction, b: represent the stride in vertical direction
        if len(stride_shape) == 1:
            stride_shape = (stride_shape[0], stride_shape[0])
        self.stride_shape = stride_shape
        # define the size of kernel (c, m, n) c is channel, m*n is convolution range
        # if input kernel is (c, m), generalize it into (c, m, 1)
        if len(convolution_shape) == 2:
            convolution_shape = (convolution_shape[0], convolution_shape[1], 1)
        self.input_tensor = None
        self._optimizer = None
        self._optimizer_weights = None
        self._optimizer_bias = None
        self.H = num_kernels
        self.C = convolution_shape[0]
        self.M = convolution_shape[1]
        self.N = convolution_shape[2]
        self.B = None
        self.X = None
        self.Y = None
        self._gradient_weights = None
        self._gradient_bias = None
        self.weights = np.random.rand(self.H, self.C, self.M, self.N)
        self.bias = np.random.rand(self.H, 1)

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

    # forward
    def forward(self, input_tensor):
        # input_tensor shape (B, C, X, Y). C is the num_channels, B is the num_batches
        # padded_input_tensor shape (B, C, X + 2 * floor(M/2) )
        # weight shape (H, C, M, N). H is the num_kernels
        # output shape (B, H, X', Y')
        # X' = 1 + (X + 2 * floor(M/2) - M)/stride.X
        # Y' = 1 + (Y + 2 * floor(N/2) - N)/stride.Y

        # generalization input_tensor
        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=3)

        # store input_tensor
        self.input_tensor = input_tensor

        self.B = input_tensor.shape[0]
        self.X = input_tensor.shape[2]
        self.Y = input_tensor.shape[3]

        # output_tensor (for sub-sampling)
        num_x = math.ceil(self.X / self.stride_shape[0])
        num_y = math.ceil(self.Y / self.stride_shape[1])
        output_tensor = np.zeros((self.B, self.H, self.X, self.Y))
        output_tensor_sub = np.zeros((self.B, self.H, num_x, num_y))
        # do convolution, stride-shape is 1
        for b in range(self.B):
            for h in range(self.H):
                for c in range(self.C):
                    output_tensor[b, h, :, :] += sgl.correlate(input_tensor[b, c, :, :], self.weights[h, c, :, :], mode='same')

                # add bias
                output_tensor[b, h, :, :] = output_tensor[b, h, :, :] + self.bias[h]

                # do sub-sampling for stride-shape isn't 1
                output_tensor_sub[b, h, :, :] = output_tensor[b, h, ::self.stride_shape[0], ::self.stride_shape[1]]

        # if 1D case, change output from (B, H, X', 1) to (B, H, X')
        if output_tensor_sub.shape[3] == 1:
            output_tensor_sub = np.reshape(output_tensor_sub, (self.B, self.H, output_tensor_sub.shape[2]))

        return output_tensor_sub

    # property optimizer
    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_weights = copy.deepcopy(self._optimizer)
        self._optimizer_bias = copy.deepcopy(self._optimizer)

    optimizer = property(get_optimizer, set_optimizer)

    def backward(self, error_tensor):
        # input_tensor
        input_tensor = copy.deepcopy(self.input_tensor)
        # generalization error_tensor
        if len(error_tensor.shape) == 3:
            error_tensor = np.expand_dims(error_tensor, axis=3)

        # upsampling error_tensor in shape (B, H, X, Y)
        error_tensor_pad = np.zeros((self.B, self.H, self.X, self.Y))
        for i in range(error_tensor.shape[2]):
            for j in range(error_tensor.shape[3]):
                error_tensor_pad[:, :, i * self.stride_shape[0], j * self.stride_shape[1]] = error_tensor[:, :, i, j]

        # rotate weight in 180 degree
        weights_rotation = np.zeros_like(self.weights)
        for h in range(self.H):
            for c in range(self.C):
                weights_rotation[h, c, :, :] = np.rot90(self.weights[h, c, :, :], 2)

        # get error_tensor for next layer. Shape is (B, C, X, Y)
        output_tensor = np.zeros((self.B, self.C, self.X, self.Y))
        for b in range(self.B):
            for c in range(self.C):
                for h in range(self.H):
                    output_tensor[b, c, :, :] += sgl.correlate(error_tensor_pad[b, h, :, :], weights_rotation[h, c, :, :], mode='same')

        # update weights and bias
        # zero padding the input_tensor(considering the size of kernel can be odd or even)
        padding_x_left = math.floor((self.M / 2))
        padding_x_right = math.floor(((self.M - 1) / 2))
        padding_y_up = math.floor((self.N / 2))
        padding_y_down = math.floor(((self.N - 1) / 2))
        input_tensor_pad = np.pad(input_tensor, ((0, 0), (0, 0), (padding_x_left, padding_x_right), (padding_y_up, padding_y_down)),
                                  mode="constant", constant_values=0)

        # get gradient weights & bias and update
        self._gradient_weights = np.zeros((self.H, self.C, self.M, self.N))
        self._gradient_bias = np.zeros((self.H, 1))
        for h in range(self.H):
            for b in range(self.B):
                for c in range(self.C):
                    self._gradient_weights[h, c, :, :] += sgl.correlate(input_tensor_pad[b, c, :, :], error_tensor_pad[b, h, :, :], mode='valid')
                self._gradient_bias[h] += np.sum(error_tensor[b, h, :, :])

        if self._optimizer is not None:
            self.weights = self._optimizer_weights.calculate_update(self.weights, self._gradient_weights)
            self.bias = self._optimizer_bias.calculate_update(self.bias, self._gradient_bias)

        # if 1D case, change output from (B, C, X', 1) to (B, C, X')
        if output_tensor.shape[3] == 1:
            output_tensor = np.reshape(output_tensor, (self.B, self.C, output_tensor.shape[2]))
        return output_tensor

    # initialize weights and bias
    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.C * self.M * self.N
        fan_out = self.H * self.M * self.N
        self.weights = weights_initializer.initialize(self.weights.shape, fan_in, fan_out)
        self.bias = bias_initializer.initialize(self.bias.shape, fan_in, fan_out)

