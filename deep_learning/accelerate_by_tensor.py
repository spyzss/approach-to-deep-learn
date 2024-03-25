import torch
import math
import numpy as np
from d2l import torch as d2l
import matplotlib.pyplot as plt

def normals(x, mu, sigma):
    p = 1 / math.sqrt(2 * math.pi * sigma ** 2)
    return p * np.exp(-0.5 / sigma ** 2 * (x - mu) ** 2) 

x = np.arange(-7,7,0.01)
params = [(0,1),(0,2),(3,1)]
d2l.plot(x,[normals(x, mu, sigma) for mu,sigma in params]
         ,xlabel = 'x' ,ylabel = 'y',figsize=(4.5,2.5),
         legend=[f'mean {mu} ,std{sigma} ' for mu,sigma in params])
print ("1")
'''

import matplotlib.pyplot as plt要引入这个库·    
x: 一个数组,表示要绘制的数据点的横坐标(x轴)
y: 一个由多个数组组成的列表,每个数组对应一条曲线的纵坐标(y轴值)
xlabel: x轴标签,通常写明该轴代表什么
ylabel: y轴标签
figsize: 确定绘图区域的大小,是一个元组(width, height)
legend: 一个由字符串组成的列表,每个字符串是对应曲线的标签,将显示在图例中
'''
plt.show()