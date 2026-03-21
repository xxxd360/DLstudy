# 任务三
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import matplotlib.pyplot as plt
# 导包
# 图像处理
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# 激活函数
import torch.nn as nn
import torch.nn.functional as F


batch_size = 64
transform = transforms.Compose([
    # 转张量
    transforms.ToTensor(),
    # 标准化
    transforms.Normalize(0.1307, 0.3081)
])
# 数据的获取与读取
train_dataset = datasets.FashionMNIST(root='D:\pytorch.pycharm\dataset',
                               train=True,
                               download=False,
                               transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset = datasets.FashionMNIST(root='D:\pytorch.pycharm\dataset',
                              train=False,
                              download=False,
                              transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)



class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=1)

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=5, padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=1)
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=3, padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=3, padding=1)

        self.branch_pool = nn.Conv2d(in_channels, 24, kernel_size=1)

    def forward(self, x):
        branch1x1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)

        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        branch_pool = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = self.branch_pool(branch_pool)

        outputs = [branch1x1, branch5x5, branch3x3, branch_pool]
        return torch.cat(outputs, dim=1)


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(88, 20, kernel_size=5)

        self.incep1 = InceptionA(in_channels=10)
        self.incep2 = InceptionA(in_channels=20)

        self.mp = nn.MaxPool2d(2)
        self.fc = nn.Linear(1408, 12)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x

# class ResidualBlock(nn.Module):
#     def __init__(self, in_channels):
#         super(ResidualBlock, self).__init__()
#         # 这里需根据实际残差块结构补充，示例中假设包含卷积、激活等操作
#         self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
#         self.relu = nn.ReLU(inplace=True)
#
#     def forward(self, x):
#         residual = x
#         out = self.conv1(x)
#         out = self.relu(out)
#         out = self.conv2(out)
#         out += residual
#         out = self.relu(out)
#         return out
#
#
# class Net(nn.Module):
#      def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 16, kernel_size=5)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
#         self.mp = nn.MaxPool2d(2)
#         self.rbblock1 = ResidualBlock(16)
#         self.rbblock2 = ResidualBlock(32)
#         self.fc = nn.Linear(512, 11)
#
#      def forward(self, x):
#         in_size = x.size(0)
#         x = self.mp(F.relu(self.conv1(x)))
#         x = self.rbblock1(x)
#         x = self.mp(F.relu(self.conv2(x)))
#         x = self.rbblock2(x)
#         x = x.view(in_size, -1)
#         x = self.fc(x)
#         return x
#

# class MixedNet(nn.Module):
#     def __init__(self):
#         super(MixedNet, self).__init__()
#         self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
#         self.inception1 = InceptionA(in_channels=20)
#         self.mp1 = nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(88, 20, kernel_size=5)
#         self.inception2 = InceptionA(in_channels=20)
#         self.rbblock = ResidualBlock(88)
#         self.mp2 = nn.MaxPool2d(2)
#         self.fc1 = nn.Linear(1408, 11)
#
#     def forward(self, x):
#         in_size = x.size(0)
#         x = self.mp1(F.relu(self.conv1(x)))
#         x = self.inception1(x)
#         x = self.mp2(F.relu(self.conv2(x)))
#         x = self.inception2(x)
#         x = self.rbblock(x)
#         x = x.view(in_size, -1)
#         x = F.relu(self.fc1(x))
#         return x


model = Net()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.5)
model.to(device)


# 训练函数封装
def train(epoch):
    running_loss = 0.0
    correct = 0
    total = 0
    # 批次       数据
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device),target.to(device)
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total     # 准确率
        if batch_idx % len(train_loader) == len(train_loader)-1:
            print('[%-5d, %-5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / len(train_loader)), 'Accuracy on batch: %.3f %%' % accuracy)
            epoch_loss = running_loss / len(train_loader)
            return epoch_loss



def test():
    correct = 0
    total = 0
    # 关闭梯度计算
    with torch.no_grad():
        for data in test_loader:
            inputs,  targets= data
            inputs,  targets = inputs.to(device),targets.to(device)
            outputs = model(inputs)
            #       判断            求最大值                 第二个维度
            _, predicted = torch.max(outputs.data, dim=1)
            total += targets.size(0)
            # 布尔判断
            correct += (predicted == targets).sum().item()
    print('Accuracy on test set: %d %%' % (100 * correct / total))


# 这样既可以作为脚本运行也可以作为模块
# 训练与测试
# if  __name__ == '__main__':
#     for epoch in range(10):
#         train(epoch)
#         test()

if __name__ == '__main__':
    loss_list = []
    for epoch in range(10):
        epoch_loss = train(epoch)
        loss_list.append(epoch_loss)
    test()

plt.plot(range(1, len(loss_list) + 1), loss_list)
plt.title('Loss per epoch')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.show()
