import torch
import math
import numpy as np
from d2l import torch as d2l
import matplotlib.pyplot as plt

def synthetic_data(w , b , num_example):
    x = torch.normal(0 , 1 , (num_example , len(w)))
    """
    均值（mean)为0， 标准差（std）为1
     (num_example, len(w))是一个元组,表示生成张量的形状(shape)。其中:
    num_example是一个整数,表示样本的数量。
    len(w)是一个整数,可能表示特征向量的长度。
    """
    y = torch.matmul(x, w) + b
    #矩阵乘法之后会给每个位置的值都加上b
    y += torch.normal(0 , 0.01 , y.shape)
    '''这个数据生成中认为y = wx + b + e e就是噪声，认为是符合std为0.01，mean=0的正态分布噪声'''
    return x , y.reshape((-1,1)) 


true_w = torch.tensor([2 , -3.4])
true_b = 4.2
feature , labels = synthetic_data(true_w , true_b , 1000)
'''用w和b这两个生成一个2维->一维的一个线性回归model'''
d2l.set_figsize()
d2l.plt.scatter(feature[:, (1)].detach().numpy() , labels.detach().numpy() , 2)
'''feature[:, (1)] 表示从 feature 张量中选择所有行的第二列(索引为1)数据。

.detach().numpy() 将张量从PyTorch的格式转换为NumPy数组格式,以便进行绘图。
labels.detach().numpy() 同样将 labels 张量转换为NumPy数组格式。
加detach是为了储存的时候不带上梯度，节约内存和计算时间

d2l.plt.scatter() 是绘制散点图的函数,它采用三个参数:
第一个参数是要绘制的 x 坐标数据(来自 feature[:, (1)])
第二个参数是要绘制的 y 坐标数据(来自 labels)
第三个参数 1 可能是指定散点的大小或颜色(具体取决于库的实现)。
'''
plt.show()