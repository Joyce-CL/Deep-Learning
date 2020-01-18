import numpy as np


class L1_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        L1_gradient = self.alpha * np.sign(weights)
        return  L1_gradient

    def norm(self, weights):
        L1_norm = self.alpha * np.sum(np.abs(weights))
        return L1_norm


class L2_Regularizer:
    def __init__(self, alpha):
        self.alpha = alpha

    def calculate_gradient(self, weights):
        L2_gradient = self.alpha * weights
        return L2_gradient

    def norm(self, weights):
        L2_norm = self.alpha * np.sqrt(np.sum(np.square(weights)))
        return L2_norm