import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)  # 输入通道数1,输出通道数16,卷积核3x3
        self.conv2 = nn.Conv2d(4, 16, 3, padding=1) # 输入通道数16,输出通道数32,卷积核3x3
        self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)  # 2x2最大池化层
        self.fc1 = nn.Linear(2880, 120)  # 全连接层
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)  # 输出维度为2

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = x.view(-1, 2880)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# 实例化模型
model = ConvNet()

# 定义输入数据和目标输出
images = torch.randn(10, 1, 28, 28)  # 模拟输入数据,batch_size=10,通道数=1,大小28x28
y_target = torch.randn(10, 2)  # 目标输出数据,形状为(10, 2)

criterion = nn.MSELoss()  # 误差损失函数
optimizer = optim.SGD(model.parameters(), lr=0.001)  # 优化器

for i in range(1000):
    optimizer.zero_grad()
    y = model(images)  # 前向传播
    loss = criterion(y, y_target)  # 计算损失
    loss.backward()  # 反向传播
    optimizer.step()  # 更新权重
    print("loss", loss.item())