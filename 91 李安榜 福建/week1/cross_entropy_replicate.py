import torch
import torch.nn as nn
import numpy as np

ce_loss = nn.CrossEntropyLoss()

a = np.array([[0.5, 0.9, 0.1],
              [0.5, 0.1, 0.9],
              [0.9, 0.1, 0.5]])

torch_a = torch.FloatTensor(a)

true_a = [1, 2, 1]
torch_true_a = torch.LongTensor(true_a)
print(ce_loss.forward(torch_a, torch_true_a))


def softmax(pred_array):
    return np.exp(pred_array) / np.sum(np.exp(pred_array), axis=1, keepdims=True)


class DiyCrossEntropy():
    def __init__(self, pred_array, true_array):
        self.pred_array = pred_array
        self.true_array = true_array

    def forward(self):
        pred = softmax(self.pred_array)
        out = torch.zeros(3, 3).scatter(dim=1, index=torch.LongTensor(self.true_array).view(-1, 1),
                                        value=1)  # torch one-hot编码方法
        # print(out.numpy())
        # print(np.log(pred))
        # print(-np.sum(out.numpy() * np.log(pred), axis=1))
        return sum(-np.sum(out.numpy() * np.log(pred), axis=1)) / self.pred_array.shape[0]


print('======================')
diy_ce = DiyCrossEntropy(a, true_a)
print(diy_ce.forward())
