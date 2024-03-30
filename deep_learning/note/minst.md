# minst手写识别实现
现在来进行最经典的minst手写识别问题的实现

导入包就不写了
## 读取数据集
~~~
trans = transforms.ToTensor()
#将图像数据转化为tensor，并且会有归一化，即除以255使得所有像素的数值均在0～1之间
mnist_train = torchvision.datasets.FashionMNIST(
root="../data", train=True, transform=trans, download=True)
mnist_test = torchvision.datasets.FashionMNIST(
root="../data", train=False, transform=trans, download=True)
~~~
这里我们用的是Fashion‐MNIST数据集，这个数据更强
这里test和train的数据都是给你处理好的，输出len(minst_train)可以看到是60000

这个数据集包含：t‐shirt、trouser、pullover、dress、coat、sandal、shirt、sneaker、bag、ankle boot，这些是标签，所以我们需要把这些标签和数字索引进行转换

## 索引转换

