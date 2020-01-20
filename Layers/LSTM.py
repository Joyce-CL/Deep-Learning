import numpy as np
import copy
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
        self.b_xh = bias_xh  # (1 x 4H)
        self.w_y = weights_y  # (k x H)
        self.b_y = bias_y  # (1 x k)

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
        self.con_tensor_xh = None
        self.out_hidden_state = None
        self.tan = [TanH() for _ in range(2)]
        self.sig = [Sigmoid() for _ in range(4)]

    def forward(self, input_tensor, hidden_state, cell_state):
        # reshape input_tensor from (j,) to (1 x j)
        input_tensor = input_tensor.reshape(1, -1)
        self.cell_state = cell_state
        self.hidden_state = hidden_state
        self.con_tensor_xh = np.concatenate((input_tensor, hidden_state), axis=1)  # shape (1 x (j + H))
        xh_wb = np.dot(self.con_tensor_xh, self.w_xh.T) + self.b_xh  # shape (1 x 4H)
        self.f_t = self.sig[0].forward(xh_wb[:, : self.H])
        self.i_t = self.sig[1].forward(xh_wb[:, self.H: 2 * self.H])
        self.c_hat_t = self.tan[0].forward(xh_wb[:, 2 * self.H: 3 * self.H])  # (1 x H)
        self.o_t = self.sig[2].forward(xh_wb[:, 3 * self.H: 4 * self.H])  # (1 x H)
        out_cell_state = self.f_t * cell_state + self.i_t * self.c_hat_t  # (1 x H)
        self.a_t = self.tan[1].forward(out_cell_state)  # (1 x H)
        out_hidden_state = self.o_t * self.a_t  # (1 x H)
        self.out_hidden_state = out_hidden_state
        output_tensor = self.sig[3].forward(np.dot(out_hidden_state, self.w_y.T) + self.b_y)  # (1 x H) * (H x k ) + (1 x k) = (1 x k)
        return output_tensor, out_hidden_state, out_cell_state

    def backward(self, error_tensor, e_hidden_state, e_cell_state):
        # get error_tensor for next layers
        error_tensor = self.sig[3].backward(error_tensor)
        e_yh = np.dot(error_tensor.reshape(1, -1), self.w_y) + e_hidden_state  # (1 x k) * (k x H) + (1 x H) = (1 x H)
        c_h = e_cell_state + self.tan[1].backward(e_yh * self.o_t)
        out_e_cell_state = c_h * self.f_t
        h1 = self.sig[0].backward(c_h * self.cell_state)  # (1 x H)
        h2 = self.sig[1].backward(c_h * self.c_hat_t)
        h3 = self.tan[0].backward(c_h * self.i_t)
        h4 = self.sig[2].backward(e_yh * self.a_t)
        con_h = np.concatenate((h1, h2, h3, h4), axis=1)  # (1 x 4H)
        out_con_tensor = np.dot(con_h, self.w_xh)  # (1 x 4H) * (4H X (j + H))
        output_tensor = out_con_tensor[:, :self.j]
        out_e_hidden_state = out_con_tensor[:, self.j:]

        # update trainable parameters (w_xh, b_xh, w_y, b_y)
        dw_xh = np.dot(con_h.T, self.con_tensor_xh)  # (1 x 4H).T * (1 X (j + H)) = (4H x (j + H))
        db_xh = con_h  # (1 x 4H)
        dw_y = np.dot(error_tensor.T, self.out_hidden_state)  # (k x H)
        db_y = error_tensor  # (1 x k)

        return output_tensor, out_e_hidden_state, out_e_cell_state, dw_xh, db_xh, dw_y, db_y


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
        # optimizer
        self._optimizer = None
        self._optimizer_w_xh = None
        self._optimizer_b_xh = None
        self._optimizer_w_y = None
        self._optimizer_b_y = None

        # initialize gradient_weights
        self.g_weights = None

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

    def backward(self, error_tensor):
        b = error_tensor.shape[0]
        # initialize output_tensor, e_hidden_state, out_e_cell_state for next layer
        output_error = np.zeros((b, self.j))
        out_e_hidden_state = np.zeros((1, self.H))
        out_e_cell_state = np.zeros((1, self.H))

        # initialize gradient dw_xh, db_xh, dw_y, db_y for trainable parameter updating
        gw_xh = np.zeros_like(self.w_xh)  # (4H x (j + H))
        gb_xh = np.zeros_like(self.b_xh)  # (1 x 4H)
        gw_y = np.zeros_like(self.w_y)  # (k x H)
        gb_y = np.zeros_like(self.b_y)  # (1 x k)

        for j in reversed(range(b)):
            output_error[j, :], out_e_hidden_state, out_e_cell_state, dw_xh, db_xh, dw_y, db_y = \
                self.layer[j].backward(error_tensor[j, :], out_e_hidden_state, out_e_cell_state)
            # sum gradient
            gw_xh += dw_xh
            gb_xh += db_xh
            gw_y += dw_y
            gb_y += db_y

        # gradient_weights for test
        self.g_weights = np.concatenate((gw_xh, gb_xh.T), axis=1)

        # updating
        if self._optimizer is not None:
            self.w_xh = self._optimizer_w_xh.calculate_update(self.w_xh, gw_xh)
            self.b_xh = self._optimizer_b_xh.calculate_update(self.b_xh, gb_xh)
            self.w_y = self._optimizer_w_y.calculate_update(self.w_y, gw_y)
            self.b_y = self._optimizer_b_y.calculate_update(self.b_y, gb_y)
        return output_error

    # property gradient_weights
    @property
    def gradient_weights(self):
        return self.g_weights

    # property memorize
    def get_memorize(self):
        return self._memorize

    def set_memorize(self, memorize):
        self._memorize = memorize

    memorize = property(get_memorize, set_memorize)

    # property weight
    def get_weights(self):
        # w_xh shape (4H x (j + H))
        # b_xh shape (1 x 4H)
        # weights shape should be (4H x (j + H + 1)
        return np.concatenate((self.w_xh, self.b_xh.T), axis=1)

    def set_weights(self, w):
        if w is None:
            self.w_xh = None
            self.b_xh = None
        else:
            self.w_xh = w[:, :-1]
            self.b_xh = w[:, -1].reshape(1, -1)

    weights = property(get_weights, set_weights)

    # property optimizer
    def get_optimizer(self):
        return self._optimizer

    def set_optimizer(self, optimizer):
        self._optimizer = optimizer
        self._optimizer_w_xh = copy.deepcopy(self._optimizer)
        self._optimizer_b_xh = copy.deepcopy(self._optimizer)
        self._optimizer_w_y = copy.deepcopy(self._optimizer)
        self._optimizer_b_y = copy.deepcopy(self._optimizer)

    optimizer = property(get_optimizer, set_optimizer)

    # calculate sum of weights in each layer for regularization
    @property
    def calculate_regularization_loss(self):
        sum_weight = 0
        if self._optimizer is not None:
            if self._optimizer.regularizer is not None:
                sum_weight = self._optimizer_w_xh.regularizer.norm(self.w_xh) + \
                             self._optimizer_w_y.regularizer.norm(self._optimizer_w_y)
        return sum_weight

    # initialize weights and bias
    def initialize(self, weights_initializer, bias_initializer):
        # fan_in is (j + H), fan_out is H
        self.w_xh = weights_initializer.initialize(self.w_xh.shape, self.j + self.H, self.H)
        self.b_xh = bias_initializer.initialize(self.b_xh.shape, self.j + self.H, self.H)
        # fan_in is H, fan_out is k
        self.w_y = weights_initializer.initialize(self.w_y.shape, self.H, self.k)
        self.b_y = bias_initializer.initialize(self.b_y.shape, self.H, self.k)
