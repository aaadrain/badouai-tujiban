import numpy as np
import torch.nn as nn
import torch

'''
y = xA^T + b
'''

# torch.nn performance
input_list = [[1, 1, 1]]
tensor_input = torch.FloatTensor(input_list)
print(tensor_input, tensor_input.shape)
linear = nn.Linear(3, 4, bias=False)
weights = linear.state_dict()['weight'].numpy()

print('weight:\n', weights, '\ntorch.nn结果：\n', linear(tensor_input).shape, linear(tensor_input).data.numpy(), '\n',
      '-' * 10)

# numpy performance
print(input_list, '\n', weights, '\n', weights.shape, '\n', weights.T, '\n', weights.T.shape)
print('np 结果：', np.sum(input_list * weights, axis=1))
print('np 结果：', np.dot(input_list, weights.T))
