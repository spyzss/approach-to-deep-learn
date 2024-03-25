import torch
import math
import numpy as np
from d2l import torch as d2l
import matplotlib.pyplot as plt
import random
from torch.utils import data
from torch import nn

def synthetic_data(w , b , num_example):
    x = torch.normal(0 , 1 , (num_example , len(w)))
    """
    均值（mean)为0， 标准差（std）为1
     (num_example, len(w))是一个元组,表示生成张量的形状(shape)。其中:
    num_example是一个整数,表示样本的数量。
    len(w)是一个整数,可能表示特征向量的长度。
    """
    y = torch.matmul(x, w) + b
    y += torch.normal(0 , 0.01 , y.shape)
    '''这个数据生成中认为y = wx + b + e e就是噪声，认为是符合std为0.01，mean=0的正态分布噪声'''
    return x , y.reshape((-1,1)) 

def load_array(data_arrays , batch_size , is_train = True):
    dataset = data.TensorDataset(*data_arrays)
    '''这个*的作用是解包
        举个例子，如果你传递了两个数组a和b给load_array函数，*data_arrays将把它们解包，
        使得函数内部实际接收的参数是data.TensorDataset(a, b)，而不是data.TensorDataset([a, b])
        这样就能够正确地构造TensorDataset对象了。
    '''
    return data.DataLoader(dataset , batch_size , shuffle=is_train)
#最后一个参数代表的是将数据打乱

true_w = torch.tensor([2 , -3.4])
true_b = 4.2
batch_size = 10
feature , labels = synthetic_data(true_w , true_b , 1000)
data_iter = load_array((feature , labels) , batch_size , True)


net = nn.Sequential(nn.Linear(2,1))
#是一个用于构建神经网络模型的容器。它允许你按照顺序将各种神经网络层组合在一起，形成一个序列化的模型
'''eg
    model = nn.Sequential(
    nn.Linear(784, 256),  # 输入层到隐藏层的线性变换
    nn.ReLU(),             # ReLU激活函数
    nn.Linear(256, 10)     # 隐藏层到输出层的线性变换
    )
'''
net[0].weight.data.normal_(0,0.01)
net[0].bias.data.fill_(0)
"""net[0]代表选中网络中的第一层
    weight and bias是linear中自带的两个参数
"""
loss = nn.SmoothL1Loss()
trainer = torch.optim.SGD(net.parameters(),lr = 0.03)

epochs = 4
for epoch in range(epochs):
    for x,y in data_iter:
        l = loss(net(x) , y)
        trainer.zero_grad()
        l.backward()
        trainer.step()
    l = loss(net(feature) , labels)
    print(l)