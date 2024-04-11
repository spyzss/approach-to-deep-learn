import torch
from torch import nn
from d2l import torch as d2l
if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    batch_size = 256
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size) 

    num_inputs, num_outputs, num_hiddens = 784, 10, 256

    w1 = nn.Parameter(torch.randn(num_inputs , num_hiddens, 
                                requires_grad= True))
    #这里的parameter是torch中的一个类，可以用来构造一个可以训练的tensor
    b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
    w2 = nn.Parameter(torch.randn(num_hiddens, num_outputs
                                , requires_grad = True))

    b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))      
    params = [w1, b1, w2, b2]
    #将四个参数化成一个列表，可能为了方便之后的操作

    def relu_re(x):
        y = torch.zeros_like(x)
        return torch.max(x,y)
    #手动实现relu函数

    net = nn.Sequential( nn.Flatten(),
                        nn.Linear(784,256),
                        nn.ReLU(),
                        nn.Linear(256,10))
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)
    init_weights(net)
    #这个就是对参数初始化的操作，不初始化w会满足高斯分布，b会是0

    batch_size, lr, num_epochs = 256, 0.1, 10
    loss = nn.CrossEntropyLoss(reduction='none')
    trainer = torch.optim.SGD(net.parameters(), lr=lr)
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)