import torch
from torch import nn
from d2l import torch as d2l
import multiprocessing

if __name__ == '__main__':
    multiprocessing.freeze_support()
    # 其他代码

    def drop_out(X , drop_rate):
        assert 0 <= drop_rate <= 1
        if drop_rate == 1:
            return torch.zeros_like(X)
        if drop_rate == 0:
            return X
        mask = (torch.rand(X.shape) > drop_rate).float()
        return mask * X / (1 - drop_rate)

    drop1 = 0.2
    drop2 = 0.5

    class net(nn.Module):
        def __init__(self, num_inputs, num_outputs, num_hiddens1, num_hiddens2,
                    is_training = True):
            super(net,self).__init__()
            #调用Mudule的init去调用父辈的构造函数
            self.num_inputs = num_inputs
            self.training = is_training
            #如果is_training=True,模型将处于训练模式,否则处于评估模式
            self.lin1 = nn.Linear(num_inputs, num_hiddens1)
            self.lin2 = nn.Linear(num_hiddens1, num_hiddens2)
            self.lin3 = nn.Linear(num_hiddens2, num_outputs)
            self.relu = nn.ReLU()
        def train_itself(self, x):
            h1 = self.lin1(x.reshape(-1 , num_inputs))
            h1 = self.relu(h1)
            if self.training == True:
                h1 = drop_out(h1 , drop1)
                # 在第一个全连接层之后添加一个dropout层
            h2 = self.lin2(h1.reshape(-1 , num_hiddens1))
            h2 = self.relu(h2)
            if self.training == True:
                h2 = drop_out(h2 , drop2)
                # 在第二个全连接层之后添加一个dropout层
            out = self.lin3(h2)
            return out

    num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
    my_net = net(num_inputs, num_outputs, num_hiddens1, num_hiddens2)
    num_epochs, lr, batch_size = 20, 0.5, 256
    loss = nn.CrossEntropyLoss(reduction='none')
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    trainer = torch.optim.SGD(my_net.parameters(), lr=lr)

    def train_my(data_iter , epochs):
        for i in  range(epochs):
            my_net.training = True
            for x, y in data_iter:
                y_hot = my_net.train_itself(x)
                l = loss(y_hot , y)
                trainer.zero_grad()
                l.mean().backward()
                trainer.step()


    train_my(train_iter , num_epochs)
    correct = 0
    total = 0
    with torch.no_grad():
        my_net.training = False
        for data, target in test_iter:
            output = my_net.train_itself(data)
            _, predicted = torch.max(output.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    print(f'Accuracy on test set: {100 * correct / total:.2f}%')
    #d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)