import torch
from torch import nn
from d2l import torch as d2l
batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) 

num_inputs, num_outputs, num_hiddens = 784, 10, 256

w1 = nn.Parameter(torch.randn(num_inputs , num_hiddens, 
                              requires_grad= True))
#这里的parameter是torch中的一个类，可以用来构造一个可以训练的tensor
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
w2 = nn.Parameter(torch.randn(num_outputs , num_hiddens 
                              , requires_grad = True))

b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))      
params = [w1, b1, w2, b2]
#将四个参数化成一个列表，可能为了方便之后的操作

def relu_re(x):
    y = torch.zeros_like(x)
    return torch.max(x,y)
#手动实现relu函数

def net(x):
    x = x.reshape((-1 , num_inputs))
    h = relu_re(x@w1 + b1)
    #这里面的@和matmul都是执行矩阵乘法的操作，但是有一定的区别
    # matmul相对更加自由，对于不和矩阵乘法规则的运算直接就用广播原理去计算了，
    #但是@是一定要符合矩阵乘法的
    return (h@w2 + b2)

loss = nn.CrossEntropyLoss(reduction='none')
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(params, lr=lr)
d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)