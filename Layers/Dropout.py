import numpy as np
from Layers.Base import Base


class Dropout(Base):
    def __init__(self, probability):
        super().__init__()
        self.probability = probability
        self.dropout_matrix = None

    def forward(self, input_tensor):
        self.dropout_matrix = np.random.binomial(1, self.probability, np.shape(input_tensor))
        if self.phase == "train":
            output_tensor = np.divide(1, self.probability) * self.dropout_matrix * input_tensor
        else:
            output_tensor = input_tensor
        return output_tensor

    def backward(self, error_tensor):
        output_tensor = self.dropout_matrix * error_tensor
        return output_tensor
