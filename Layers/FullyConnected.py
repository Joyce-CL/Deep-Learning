import numpy as np

class FullyConnected:

    def __init__(self, input_size, output_size):
        # In follow comment write input_size = j, output_size = k, batch_size = b
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(self.output_size, self.input_size + 1)
        self._optimizer = None
        self.input_tensor = None
        self.gradient_tensor = None

    def forward(self, input_tensor):
        # Z = A * W.T with
        # A.shape = b * (j + 1)
        # W.T.shape = (j + 1) * k
        # Z.shape = b * k
        self.input_tensor = input_tensor
        all_one_vector = np.ones([input_tensor.shape[0], 1], 'int')  # batch_size = input_tensor.shape[0]
        added_input_tensor = np.concatenate((input_tensor, all_one_vector), axis=1)
        input_tensor = added_input_tensor.dot(self.weights.T)
        return input_tensor

    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer

    optimizer = property(get_optimizer, set_optimizer)

    def backward(self, error_tensor):
        # error_tensor.shape = b * k (En')
        tmp = error_tensor
        error_tensor_with_bias = error_tensor.dot(self.weights)   # shape = b * (j + 1)
        error_tensor = np.delete(error_tensor_with_bias, -1, axis=1)
        # update
        # W't+1 = W't - learning_rate * X'T * En' (gradient_tensor = X'T * En')
        # X'.shape = b * (j + 1) (input_tensor)
        # W't+1.shape = W't.shape = (j + 1) * k
        # But self.weights.shape in the program is k * (j + 1), so gradient_tensor = X'T * En' need to be transposed
        all_one_vector = np.ones([self.input_tensor.shape[0], 1], 'int')
        added_input_tensor = np.concatenate((self.input_tensor, all_one_vector), axis=1)
        self.gradient_tensor = np.dot(added_input_tensor.T, tmp).T  # shape = k * (j + 1)
        # Don't perform an update if the optimizer is unset
        if self._optimizer is not None:
            self.weights = self._optimizer.calculate_update(self.weights, self.gradient_tensor)
        return error_tensor

    def initialize(self, weights_initializer, bias_initializer):
        fan_in = self.input_size
        fan_out = self.output_size
        weights = weights_initializer.initialize(self.weights[:, :-1].shape, fan_in, fan_out)
        bias = bias_initializer.initialize(self.weights.shape[0], fan_in, fan_out)
        # expand dimension of bias from (3,) to (1, 3)
        bias = np.expand_dims(bias, axis=1)
        self.weights = np.concatenate((weights, bias), axis=1)

    @property
    def gradient_weights(self):
        return self.gradient_tensor
