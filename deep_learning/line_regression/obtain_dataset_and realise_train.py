import torch
import math
import numpy as np
from d2l import torch as d2l
import matplotlib.pyplot as plt
import random

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

def data_iter(batch_size , feaure , labels):
    num_example = len(feature)
    indices = list(range(num_example))
    
    #创建从0 -- numfeature - 1的列表，作为之后的索引
    random.shuffle(indices)
    #打乱这个列表的顺序，从而将数据随机化
    for i in range(0 , num_example , batch_size):
        #i从零开始，每一步步长是batch_size，直到到达或者超过num_example
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size , num_example)]
        )
        yield feature[batch_indices] , labels[batch_indices]
        #yield可以实现小批量的传递数据，不想return会直接结束这个函数
 
true_w = torch.tensor([2 , -3.4])
true_b = 4.2
feature , labels = synthetic_data(true_w , true_b , 1000)
'''用w和b这两个生成一个2维->一维的一个线性回归model'''
w = torch.normal(0 , 0.01,size = (2 , 1) , requires_grad=True)
b = torch.zeros(1 , requires_grad=True)
#初始随机两个参数，之后的目标就是更新这两个参数
def linearg(x,w,b):
    return torch.matmul(x,w) + b
#定义模型
def squared_loss(y_hat , y):
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
#定义损失函数
def sgd(params , lr , batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()

#定义梯度下降的优化算法
epochs = 3
lr = 0.03
batch_size = 10

for epoch in range(epochs):
    for x , y in data_iter(batch_size , feature , labels):
        l = squared_loss(linearg(x,w,b) , y)

        l.sum().backward()
        #对损失函数求梯度是为了最小化损失函数，使得预测值准确
        sgd([w , b] , lr , batch_size)
    with torch.no_grad():
        train_l = squared_loss(linearg(feature , w , b) , labels)
        print(w , b)


