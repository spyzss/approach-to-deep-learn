import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
from IPython import display
import matplotlib.pyplot as plt
d2l.use_svg_display()

#导入数据
trans = transforms.ToTensor()
#将图像数据转化为tensor，并且会有归一化，即除以255使得所有像素的数值均在0～1之间
mnist_train = torchvision.datasets.FashionMNIST(
root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
root="../data", train=False, transform=trans, download=True)

def get_fashion_mnist_labels(labels): 
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
    'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]

'''列表推导式的作用是将一个数字列表labels映射到text_labels列表,生成一个新的列表,其中每个元素是text_labels中对应索引位置的标签字符串。

例如,如果labels = [3, 0, 5, 7],那么这个列表推导式将返回['dress', 't-shirt', 'sandal', 'sneaker']。'''
def get_dataloader_workers():
    return 4

def load_data_fashion_mnist(batch_size, resize=None): #@save
    
    trans = [transforms.ToTensor()]
    #将图像数据转化为tensor，并且会有归一化，即除以255使得所有像素的数值均在0～1之间
    if resize:
        trans.insert(0, transforms.Resize(resize))
        #在trans最前端插入一个变换操作
        '''如果resize是一个整数,则将图像的最小边缩放到该值，而保持长宽比不变
        如果resize是一个元组,如(256, 256)，则将图像直接缩放到该分辨率大小'''
    trans = transforms.Compose(trans)
    #执行trans中的变换操作
    mnist_train = torchvision.datasets.FashionMNIST(
    root="../data", train=True, transform=trans, download=True)

    mnist_test = torchvision.datasets.FashionMNIST(
    root="../data", train=False, transform=trans, download=True)
    
    return (data.DataLoader(mnist_train, batch_size, shuffle=True,
    num_workers=get_dataloader_workers()),
    data.DataLoader(mnist_test, batch_size, shuffle=False,
    num_workers=get_dataloader_workers()))
#上面的不用看，在dataloader中有详细解释

batch_size = 256
train_iter, test_iter = load_data_fashion_mnist(batch_size)
#这个具体的细节就是dataloader中的代码了，这里直接用封装好的就OK

def softmax(x):
    x_exp = torch.exp(x)
    #每一个数字都做幂次运算
    x_exp_sum = x_exp.sum(1 , keepdim = True)
    return x_exp / x_exp_sum
#也是运用了广播机制
num_inputs = 784
num_outputs = 10
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True)
b = torch.zeros(num_outputs, requires_grad=True)

def net(x):
    #定义模型
    #w.shape[0]指的是w第一个维度的大小，即行数
    return softmax(torch.matmul(x.reshape(-1,W.shape[0]),W) + b)

def cross_entropy(y_hat, y):
    #比如说y = torch.tensor([0, 2])，代表的是第一个数据分类是0（第一个），第二个数据分类是2（第三个）
    return - torch.log(y_hat[range(len(y_hat)), y])
    #range(len(y_hat))指的是选择每一行，range(len(y_hat))，y指的是选择每一行上对应y位置的数

def accuracy(y_hat, y): #@save
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        #代表y_hat得是一个矩阵
        y_hat = y_hat.argmax(axis=1)
        #如果是矩阵，我们计算每一行的最大值，这个就是我们要的结果
        #这个函数返回的是最大值的索引，也就是预测出来是谁
    cmp = y_hat.type(y.dtype) == y
    #将y_hat转化为y的shape再去进行比较
    return float(cmp.type(y.dtype).sum())

class Accumulator: 
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n
        
    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
        #zip把这两个东西打包成元组，从而简化代码
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
        #重载[]
def evaluate_accuracy(net, data_iter):
    if isinstance(net , torch.nn.Module):
        net.eval()
        #如果这个模型是torch中自带的模型，那么就把他调成评估模式
    metric = Accumulator(2)# 正确预测数、预测总数

    with torch.no_grad():
        for x , y in data_iter:
            metric.add(accuracy(net(x),y) , y.numel())
            #y.numel代表的是y中一共有多少个数
            #这个代码实际就是执行了metric[0]+=(accuracy(net(x)), metric[1]+=(y.numel())
    return metric[0]  / metric[1]

def train_minst(net,train_iter ,   loss , updater):
    if isinstance(net , torch.nn.Module):
        net.train()
        #如果是内置的网络，就转化为训练模式
    metric = Accumulator(3)
    for x , y in train_iter:
        y_hat = net(x)
        l = loss(y_hat , y)
        if isinstance(updater , torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            #使用自己写的优化器
            l.sum().backward()
            updater(x.shape[0])

        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
        # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]


class Animator: #@save
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
    ylim=None, xscale='linear', yscale='linear',
    fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1,
    figsize=(3.5, 2.5)):
    # 增量地绘制多条线
        if legend is None:
            legend = []
        d2l.use_svg_display()
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(
        self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
        self.X, self.Y, self.fmts = None, None, fmts

    def add(self, x, y):
        # 向图表中添加多个数据点
            if not hasattr(y, "__len__"):
                y = [y]
            n = len(y)
            if not hasattr(x, "__len__"):
                x = [x] * n
            if not self.X:
                self.X = [[] for _ in range(n)]
            if not self.Y:
                self.Y = [[] for _ in range(n)]
            for i, (a, b) in enumerate(zip(x, y)):
                if a is not None and b is not None:
                    self.X[i].append(a)
                    self.Y[i].append(b)
            self.axes[0].cla()
            for x, y, fmt in zip(self.X, self.Y, self.fmts):
                self.axes[0].plot(x, y, fmt)
            self.config_axes()
            display.display(self.fig)
            display.clear_output(wait=True)

#这个是画图的代码，可以不用理解，只是为了可视化过程

lr = 0.1
def updater(batch_size):
    return d2l.sgd([W,b] , lr , batch_size)

def train(net, train_iter, test_iter, loss, num_epochs, updater):
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
    legend=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_minst(net, train_iter, loss, updater)
        test_acc = evaluate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    train_loss, train_acc = train_metrics
    assert train_loss < 0.5, train_loss
    assert train_acc <= 1 and train_acc > 0.7, train_acc
    assert test_acc <= 1 and test_acc > 0.7, test_acc


num_epochs = 10
train(net, train_iter, test_iter, cross_entropy, num_epochs, updater)
plt.show()