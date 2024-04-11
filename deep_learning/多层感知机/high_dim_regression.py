import torch
from torch import nn
from d2l import torch as d2l

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

def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2
#L2范数惩罚的实现

lr = 0.01
loss = nn.MSELoss()
net = nn.Sequential(nn.Linear(200,1))
updater = torch.optim.SGD(net.parameters(), lr=lr)

def train(lambd):
    #lambd是超参数
    epochs = 100
    for i in range(epochs):
        for x, y in train_iter:
            updater.zero_grad()
            y_train = net(x)
            
            l = loss(y_train , y) + lambd * l2_penalty(w)
            l.backward()
            updater.step()

train(1)

print(w,b)
print(true_w,true_b)
    
    