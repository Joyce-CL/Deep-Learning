import numpy as np
from Layers.Base import Base
from Layers.TanH import TanH
from Layers.Sigmoid import Sigmoid

class LSTM_cell:
    def __init__(self, weights_xh, bias_xh, weights_y, bias_y):
        self.k, self.H = weights_y.shape
        self.j = weights_xh.shape[1] - self.H
        # initialize trainable parameters
        # Weights of hidden_layer are Wf, Wi, Wc, W0, shape (H x H), 4 as shape (4H x H)
        # Weights of input_tensor are W1, W2, W3, W4, shape (H x j), 4 as shape (4H x j)
        # Combine all these weights together as shape (4H x (j + H))
        # Combine input_tensor and hidden_state as shape (1 x (j + H))
        # Weights of output_tensor is Wy, shape (k x H)
        self.w_xh = weights_xh  # W_xh = [[W1, Wf], [W2, Wi], [W3, Wc], [W4, W0]] shape(4H x (j + H))
        self.b_xh = bias_xh
        self.w_y = weights_y
        self.b_y = bias_y

        self.input_xh = None
        self.cell_state = None
        self.hidden_state = None
        self.tan = []
        self.sig = []
        # store parameter for backward
        self.f_t = None
        self.i_t = None
        self.c_hat_t = None
        self.o_t = None
        self.a_t = None
        self.cell_state = None

    def forward(self, input_tensor, hidden_state, cell_state):
        # reshape input_tensor from (j,) to (1 x j)
        input_tensor = input_tensor.reshape(1, -1)
        self.cell_state = cell_state
        self.hidden_state = hidden_state
        con_tensor = np.concatenate((input_tensor, hidden_state), axis=1)  # shape (1 x (j + H))
        xh_wb = np.dot(con_tensor, self.w_xh.T) + self.b_xh  # shape (1 x 4H)
        self.tan = [TanH() for _ in range(2)]
        self.sig = [Sigmoid() for _ in range(4)]
        self.f_t = self.sig[0].forward(xh_wb[:, : self.H])
        self.i_t = self.sig[1].forward(xh_wb[:, self.H: 2 * self.H])
        self.c_hat_t = self.tan[0].forward(xh_wb[:, 2 * self.H: 3 * self.H])  # (1 x H)
        self.o_t = self.sig[2].forward(xh_wb[:, 3 * self.H: 4 * self.H])  # (1 x H)
        out_cell_state = self.f_t * cell_state + self.i_t * self.c_hat_t  # (1 x H)
        self.a_t = self.tan[1].forward(out_cell_state)  # (1 x H)
        out_hidden_state = self.o_t * self.a_t  # (1 x H)
        output_tensor = self.sig[3].forward(np.dot(out_hidden_state, self.w_y.T) + self.b_y)  # (1 x H) * (H x k ) + (1 x k) = (1 x k)
        return output_tensor, out_hidden_state, out_cell_state

    def backward(self, error_tensor, e_hidden_state, e_cell_state):
        e_yh = np.dot(self.sig[3].backward(error_tensor), self.w_y) + e_hidden_state  # (1 x k) * (k x H) + (1 x H) = (1 x H)
        c_h = e_cell_state + self.tan[1].backward(e_yh * self.o_t)
        out_e_cell_state = c_h * self.f_t
        h1 = self.sig[0].backward(c_h * self.cell_state)  # (1 x H)
        h2 = self.sig[1].backward(c_h * self.c_hat_t)
        h3 = self.tan[0].backward(c_h * self.i_t)
        h4 = self.sig[2].backward(e_yh * self.a_t)
        # w_1 = self.w_xh[:self.H, :self.j]
        # w_2 = self.w_xh[self.H: 2 * self.H, :self.j]
        # w_3 = self.w_xh[2 * self.H: 3 * self.H, :self.j]
        # w_4 = self.w_xh[3 * self.H: 4 * self.H, :self.j]
        # w_f = self.w_xh[self.j:, :self.H]
        # w_i = self.w_xh[self.j:, self.H: 2 * self.H]
        # w_c = self.w_xh[self.j:, 2 * self.H: 3 * self.H]
        # w_0 = self.w_xh[self.j:, 3 * self.H: 4 * self.H]
        out_con_tensor = np.concatenate((h1, h2, h3, h4), axis=1) * self.w_xh  # (1 x 4H) * (4H X (j + H))
        output_tensor = out_con_tensor[:, :self.j]
        e_hidden_state = out_con_tensor[:, self.j:]
        return output_tensor, e_hidden_state, out_e_cell_state


class LSTM(Base):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        # size of input tensor is (1 x j)
        # size of hidden layer is (1 x H)
        # size of output tensor is (1 x k)
        self.j = input_size
        self.H = hidden_size
        self.k = output_size

        # initialize hidden_state and cell_state
        self.hidden_state = np.zeros((1, self.H))
        self.cell_state = np.zeros((1, self.H))
        # set memorize to decide whether the prev_hidden_state propagate to the next batch
        self._memorize = False

        # initialize trainable parameters
        # Weights of hidden_layer are Wf, Wi, Wc, W0, shape (H x H), 4 as shape (4H x H)
        # Weights of input_tensor are W1, W2, W3, W4, shape (H x j), 4 as shape (4H x j)
        # Combine all these weights together as shape (4H x (j + H))
        # Combine input_tensor and hidden_state as shape (1 x (j + H))
        # Weights of output_tensor is Wy, shape (k x H)
        self.w_xh = np.random.rand(4 * self.H, (self.j + self.H))  # W_xh = [[W1, Wf], [W2, Wi], [W3, Wc], [W4, W0]]
        self.b_xh = np.random.rand(1, 4 * self.H)
        self.w_y = np.random.rand(self.k, self.H)
        self.b_y = np.random.rand(1, self.k)
        # store different cells
        self.layer = []

    def forward(self, input_tensor):
        b = input_tensor.shape[0]
        output_tensor = np.zeros((b, self.k))
        # check memorize
        if not self._memorize:
            # not relationship between the prev_batch and the next_batch
            self.hidden_state = np.zeros((1, self.H))
            self.cell_state = np.zeros((1, self.H))
        for i in range(b):
            cell = LSTM_cell(self.w_xh, self.b_xh, self.w_y, self.b_y)
            output_tensor[i, :], self.hidden_state, self.cell_state = cell.forward(input_tensor[i, :], self.hidden_state, self.cell_state)
            self.layer.append(cell)
        return output_tensor

    # property memorize
    def get_memorize(self):
        return self._memorize

    def set_memorize(self, memorize):
        self._memorize = memorize

    memorize = property(get_memorize, set_memorize)

    # initialize weights and bias
    def initialize(self, weights_initializer, bias_initializer):
        # fan_in is (j + H), fan_out is H
        self.w_xh = weights_initializer.initialize(self.w_xh.shape, self.j + self.H, self.H)
        self.b_xh = bias_initializer.initialize(self.b_xh.shape, self.j + self.H, self.H)
        # fan_in is H, fan_out is k
        self.w_y = weights_initializer.initialize(self.w_y.shape, self.H, self.k)
        self.b_y = bias_initializer.initialize(self.b_y.shape, self.H, self.k)
