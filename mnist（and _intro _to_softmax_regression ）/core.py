import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class MyMNISTDataset(Dataset):
    def __init__(self, data_path, label_path, transform=None):
        with open(data_path, 'rb') as f:
            # 跳过前 16 个字节的文件头
            f.read(16)
            # 读取数据部分
            self.data = np.frombuffer(f.read(), dtype=np.uint8).reshape(-1, 28, 28)

        with open(label_path, 'rb') as f:
            # 跳过前 8 个字节的文件头
            f.read(8)
            # 读取标签部分
            self.labels = np.frombuffer(f.read(), dtype=np.uint8)

        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img = self.data[idx]
        label = self.labels[idx]

        if self.transform:
            img = self.transform(img)

        return img, label

# 数据转换
transform = transforms.Compose([
    transforms.ToTensor(),
])

# 训练集
train_data = r"E:\code_spy\mnist\mnist_data\t10k-images-idx3-ubyte\t10k-images.idx3-ubyte"
train_labels = r"E:\code_spy\mnist\mnist_data\t10k-labels-idx1-ubyte\t10k-labels.idx1-ubyte"
test_data = r"E:\code_spy\mnist\mnist_data\train-images-idx3-ubyte\train-images.idx3-ubyte"
test_labels = r"E:\code_spy\mnist\mnist_data\train-labels-idx1-ubyte\train-labels.idx1-ubyte"

# 创建数据集实例
train_dataset = MyMNISTDataset(train_data, train_labels, transform=transform)
test_dataset = MyMNISTDataset(test_data, test_labels, transform=transform)

# 创建数据加载器
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1000, shuffle=False)

# 定义神经网络模型
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 实例化模型
model = Net()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
learning_rate = 0.01  # 学习率

# 创建 Adam 优化器，并传入超参数
optimizer = optim.Adam(model.parameters(), lr=learning_rate)


# 训练模型
epochs = 100
for epoch in range(epochs):
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data, target in test_loader:
        output = model(data)
        _, predicted = torch.max(output.data, 1)
        total += target.size(0)
        correct += (predicted == target).sum().item()

print(f'Accuracy on test set: {100 * correct / total:.2f}%')
