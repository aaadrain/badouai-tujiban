import torch
import torch.nn as nn

import numpy as np


class TorchRnn(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(TorchRnn, self).__init__()
        self.rnn = nn.RNN(input_size=input_size, hidden_size=hidden_size, bias=False, batch_first=True)

    def forward(self, input):
        output = self.rnn(input)
        print(output)
        return output


class DiyRnn():
    '''
    h_t = tanh(W_ih·x_t + b_ih + W_hh·h_(t-1) + b_hh)
    '''

    def __init__(self, w_ih, w_hh, hidden_size):
        self.W_ih = w_ih
        self.W_hh = w_hh
        self.hidden_size = hidden_size

    def forward(self, x):
        h_t = np.zeros((self.hidden_size))
        output = []
        for xi in x:
            print('xi',xi,'xi.T',xi.T)
            ux = np.dot(self.W_ih, xi)
            wh = np.dot(self.W_hh, h_t)
            h_t = np.tanh(ux + wh)
            output.append(h_t)
        print(output)

#  输入是一组array。

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
a_shape = a.shape
hidden_size = 6
tensor_a = torch.FloatTensor([a])
print(a, tensor_a)
print(a.shape)
rnn_torch = TorchRnn(a_shape[1], hidden_size)
'''
权重初始化
'''
w_ih = rnn_torch.state_dict()['rnn.weight_ih_l0']
w_hh = rnn_torch.state_dict()['rnn.weight_hh_l0']

rnn_torch.forward(tensor_a)

rnn_diy = DiyRnn(w_ih, w_hh, hidden_size)
rnn_diy.forward(a)
