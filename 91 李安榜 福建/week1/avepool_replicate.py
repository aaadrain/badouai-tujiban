import numpy as np
import torch
import torch.nn as nn
a = [[1,2,3,4,5,6,7,],[1,2,3,4,5,3,0,]]

print(torch.Tensor([a]).squeeze().shape)
avePool = nn.AvgPool1d(4)

print(avePool(torch.Tensor([a]).transpose(1,2)))

def diy_Pool(kernel_size:int, a:np.array)-> np.array:
    pass


