import numpy as np
import math

class Pooling:
    def __init__(self, stride_shape, pooling_shape):
        self.stride_shape = stride_shape
        self.pooling_shape = pooling_shape
        self.position = None
        self.num=0

    def forward(self, input_tensor):
        # generalize input_tensor
        self.position = []
        self.num+=1
        #print(self.num)
        if len(input_tensor.shape) == 3:
            input_tensor = np.expand_dims(input_tensor, axis=3)

        self.B = input_tensor.shape[0]
        self.C = input_tensor.shape[1]
        self.X = input_tensor.shape[2]
        self.Y = input_tensor.shape[3]

        # size of output is (B, C, num_x, num_y)
        self.num_x = math.floor((self.X - self.pooling_shape[0]) / self.stride_shape[0] + 1)
        self.num_y = math.floor((self.Y - self.pooling_shape[1]) / self.stride_shape[1] + 1)
        output_tensor = np.zeros((self.B, self.C, self.num_x, self.num_y))

        for b in range(self.B):
            for c in range(self.C):
                for i in range(self.num_x):
                    for j in range(self.num_y):
                        mask = input_tensor[b, c,
                               i * self.stride_shape[0]: i * self.stride_shape[0] + self.pooling_shape[0],
                               j * self.stride_shape[1]: j * self.stride_shape[1] + self.pooling_shape[1]]
                        output_tensor[b, c, i, j] = np.max(mask)
                        max_position = np.where(mask == np.max(mask))
                        x_index = max_position[0] + i * self.stride_shape[0]
                        y_index = max_position[1] + j * self.stride_shape[1]
                        self.position.append([b, c, x_index[0], y_index[0]])

        # if 1D case, change output from (B, C, X', 1) to (B, C, X')
        if output_tensor.shape[3] == 1:
            output_tensor = np.reshape(output_tensor, (self.B, self.C, output_tensor.shape[2]))
        return output_tensor

    def backward(self, error_tensor):
        # generalize error_tensor
        if len(error_tensor.shape) == 3:
            error_tensor = np.expand_dims(error_tensor, axis=3)

        output_tensor = np.zeros((self.B, self.C, self.X, self.Y))
        tmp = np.reshape(error_tensor, (-1, 1))
        # position of interpolation
        for i in range(len(self.position)):
            output_tensor_b = self.position[i][0]
            output_tensor_c = self.position[i][1]
            output_tensor_x = self.position[i][2]
            output_tensor_y = self.position[i][3]
            output_tensor[output_tensor_b, output_tensor_c, output_tensor_x, output_tensor_y] += tmp[i]

        # if 1D case, change output from (B, C, X', 1) to (B, C, X')
        if output_tensor.shape[3] == 1:
            output_tensor = np.reshape(output_tensor, (self.B, self.C, output_tensor.shape[2]))
        return output_tensor
