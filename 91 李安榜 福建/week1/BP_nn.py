import numpy as np
import torch
import torch.nn as nn
import copy

class TorchModel(nn.Module):
    def __init__(self, input_dim):
        super(TorchModel, self).__init__()
        self.line_layer = nn.Linear(input_dim, input_dim, bias=False)
        self.activate = nn.Sigmoid()
        self.mse = nn.functional.mse_loss

    def forward(self, x, y=None):
        y_pred = self.line_layer(x)
        y_pred = self.activate(y_pred)
        if y is not None:
            loss = self.mse(y_pred, y)
            return loss
        else:
            return y_pred


class DiyModel():
    def __init__(self, weight):
        self.weight = weight

    def forward(self, x, y=None):
        y_pred = np.dot(self.weight, x)
        y_pred = self.sigmoid(y_pred)
        if y is not None:
            return self.mse(y_pred, y)
        else:
            return y_pred

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def mse(self, y_pred, y_true):
        return np.sum(np.square((y_pred - y_true))) / len(y_pred)

    def caculate_grad(self,  y_pred,y_true, x):
        '''
        链式求导法则
        '''
        mse_grad = 2 * (y_pred - y_true) / len(y_pred)
        sigmoid_grad = y_pred * (1 - y_pred)

        grad = mse_grad * sigmoid_grad

        grad = np.dot(grad.reshape(len(x), 1), x.reshape(1, len(x)))

        return grad
#梯度更新
def diy_sgd(grad, weight, learning_rate):
    return weight - grad * learning_rate


x = np.array([1, 2, 3, 4])  #输入
y = np.array([3, 2, 4, 5])  #预期输出


#torch实验
torch_model = TorchModel(len(x))
print(torch_model.state_dict())
torch_model_w = torch_model.state_dict()["line_layer.weight"]
print(torch_model_w, "初始化权重")
numpy_model_w = copy.deepcopy(torch_model_w.numpy())

torch_x = torch.FloatTensor([x])
torch_y = torch.FloatTensor([y])
#torch的前向计算过程，得到loss
torch_loss = torch_model.forward(torch_x, torch_y)
print("torch模型计算loss：", torch_loss)
# #手动实现loss计算
diy_model = DiyModel(numpy_model_w)
diy_loss = diy_model.forward(x, y)
print("diy模型计算loss：", diy_loss)



# #设定优化器
learning_rate = 0.1
optimizer = torch.optim.SGD(torch_model.parameters(), lr=learning_rate)
# optimizer = torch.optim.Adam(torch_model.parameters())
optimizer.zero_grad()
#
# #pytorch的反向传播操作
torch_loss.backward()
print(torch_model.line_layer.weight.grad, "torch 计算梯度")  #查看某层权重的梯度

print(diy_model.forward(x),'---------')
# #手动实现反向传播
grad = diy_model.caculate_grad(diy_model.forward(x), y, x)
# print(grad, "diy 计算梯度")
#
# #torch梯度更新
optimizer.step()
# #查看更新后权重
update_torch_model_w = torch_model.state_dict()["line_layer.weight"]
print(update_torch_model_w, "torch更新后权重")
#
# #手动梯度更新
diy_update_w = diy_sgd(grad, numpy_model_w, learning_rate)
# diy_update_w = diy_adam(grad, numpy_model_w)
print(diy_update_w, "diy更新权重")