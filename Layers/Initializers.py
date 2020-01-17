import numpy as np


class Constant:
    def __init__(self, constant):
        self.weight_initialization = constant

    def initialize(self, weights_shape, fan_in, fan_out):
        initialized_tensor = self.weight_initialization * np.ones(weights_shape)
        return initialized_tensor


class UniformRandom:
    def initialize(self, weights_shape, fan_in, fan_out):
        initialized_tensor = np.random.rand(weights_shape[0], weights_shape[1])
        return initialized_tensor


class Xavier:
    def initialize(self, weights_shape, fan_in, fan_out):
        theta = np.sqrt(np.divide(2, (fan_in + fan_out)))
        initialized_tensor = np.random.normal(0, theta, weights_shape)
        return initialized_tensor


class He:
    def initialize(self, weights_shape, fan_in, fan_out):
        theta = np.sqrt(np.divide(2, fan_in))
        initialized_tensor = np.random.normal(0, theta, weights_shape)
        return initialized_tensor
