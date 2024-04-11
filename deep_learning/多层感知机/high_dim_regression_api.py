
import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

#%matplotlib inline 
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones((num_inputs, 1)) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train)
train_iter = d2l.load_array(train_data, batch_size)
# load_array这个函数将数据集分为小批次，每次返回这个小批次

test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

w = torch.normal(0 , 1 , size = (num_inputs , 1) , requires_grad= True)
b = torch.zeros(1 , requires_grad= True)
#这里可不要直接写 b=0 了，这样是没法求梯度的
params = [w,b]



def train_concise(wd):
    net = nn.Sequential(nn.Linear(num_inputs, 1))
    #创建一个神经网路，里面只有一个线性层
    for param in net.parameters():
        param.data.normal_()
    #对线性层的所有参数初始化（其实也就是w和b）
    #使用均值为0、标准差为1的正态分布初始化
    loss = nn.MSELoss(reduction='none')
    num_epochs, lr = 100, 0.003
    # 偏置参数没有衰减
    trainer = torch.optim.SGD([
        {"params":net[0].weight,'weight_decay': wd},
        {"params":net[0].bias}], lr=lr)
    """
    net[0].weight是神经网络第一层(在这种情况下是唯一一层)的权重张量
    'weight_decay': wd指定了该层权重的权重衰减(weight decay)系数为wd
    """
    animator = d2l.Animator(xlabel='epochs', ylabel='loss', yscale='log',
                          xlim=[5, num_epochs], legend=['train', 'test'])
    #画图用的
    for epoch in range(num_epochs):
        for X, y in train_iter:
            trainer.zero_grad()
            l = loss(net(X), y)
            l.mean().backward()
            #因为前面有reduction=none所以要加.mean
            trainer.step()
    if (epoch + 1) % 5 == 0:
        animator.add(epoch + 1,
        (d2l.evaluate_loss(net, train_iter, loss),
         d2l.evaluate_loss(net, test_iter, loss)))
    print('w的L2范数：', net[0].weight.norm().item())
train_concise(3)
plt.show()