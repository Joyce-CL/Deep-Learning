import numpy as np

class Sgd:

    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        updated_weight = weight_tensor - self.learning_rate * gradient_tensor
        return updated_weight

class SgdWithMomentum:

    def __init__(self, learning_rate, momentum_rate):
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        weight_tensor = weight_tensor + self.v
        return weight_tensor

class Adam:

    def __init__(self, learning_rate, mu, rho):
        self.learning_rate = learning_rate
        self.mu = mu
        self.rho = rho
        self.v = 0
        self.r = 0
        self.k = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        epsilon = 1e-8
        self.k += 1
        self.v = self.mu * self.v + (1 - self.mu) * gradient_tensor
        self.r = self.rho * self.r + (1 - self.rho) * np.power(gradient_tensor, 2)

        v_hat = self.v / (1 - np.power(self.mu, self.k))
        r_hat = self.r / (1 - np.power(self.rho, self.k))

        weight_tensor = weight_tensor - self.learning_rate * np.divide((v_hat + epsilon), (np.sqrt(r_hat) + epsilon))
        return weight_tensor

