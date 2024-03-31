import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l
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

