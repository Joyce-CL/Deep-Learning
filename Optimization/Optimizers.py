import numpy as np


class Base_optimizer:

    def __init__(self):
        self.regularizer = None

    def add_regularizer(self, regularizer):
        self.regularizer = regularizer


class Sgd(Base_optimizer):

    def __init__(self, learning_rate):
        super().__init__()
        self.learning_rate = learning_rate

    def calculate_update(self, weight_tensor, gradient_tensor):
        if self.regularizer is None:
            updated_weight = weight_tensor - self.learning_rate * gradient_tensor
        # Gradient caused by regularization and gradient caused by Loss function
        else:
            updated_weight = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor) \
                         - self.learning_rate * gradient_tensor
        return updated_weight


class SgdWithMomentum(Base_optimizer):

    def __init__(self, learning_rate, momentum_rate = 0.9):
        super().__init__()
        self.learning_rate = learning_rate
        self.momentum_rate = momentum_rate
        self.v = 0

    def calculate_update(self, weight_tensor, gradient_tensor):
        self.v = self.momentum_rate * self.v - self.learning_rate * gradient_tensor
        if self.regularizer is None:
            weight_tensor = weight_tensor + self.v
        else:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor) + self.v
        return weight_tensor


class Adam(Base_optimizer):

    def __init__(self, learning_rate, mu = 0.9, rho = 0.999):
        super().__init__()
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

        if self.regularizer is None:
            weight_tensor = weight_tensor - self.learning_rate * np.divide((v_hat + epsilon), (np.sqrt(r_hat) + epsilon))
        else:
            weight_tensor = weight_tensor - self.learning_rate * self.regularizer.calculate_gradient(weight_tensor) \
                            - self.learning_rate * np.divide((v_hat + epsilon), (np.sqrt(r_hat) + epsilon))
        return weight_tensor

