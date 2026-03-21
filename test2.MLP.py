# 任务二
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
import matplotlib.pyplot as plt
# 图像处理
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
# 激活函数
import  torch.nn.functional as F
# 优化器
import torch.optim as optim

batch_size = 64
transform = transforms.Compose([
    # 转张量
    transforms.ToTensor(),
    # 标准化
    transforms.Normalize((0.1307),(0.3081))])
# 数据的获取与读取
train_dataset = datasets.FashionMNIST(root=r"D:\pytorch.pycharm\dataset",
                                train = True,
                                download=False,
                                transform=transform)
train_loader = DataLoader(train_dataset,
                          shuffle=True,
                          batch_size=batch_size)
test_dataset  = datasets.FashionMNIST(root=r"D:\pytorch.pycharm\dataset",
                                train = False,
                                download=False,
                                transform=transform)
test_loader = DataLoader(test_dataset,
                         shuffle=False,
                         batch_size=batch_size)

# 构造模块
class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.l1 = torch.nn.Linear(784, 512)
        self.l2 = torch.nn.Linear(512, 256)
        self.l3 = torch.nn.Linear(256, 128)
        self.l4 = torch.nn.Linear(128, 64)
        self.l5 = torch.nn.Linear(64, 10)

    def forward(self, x):
        x = x.view(-1, 784)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = F.relu(self.l4(x))
        return self.l5(x)

model = Net()
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(),lr=0.01,momentum=0.5)

# 训练函数封装
def train(epoch):
    running_loss = 0.0
    correct = 0
    total = 0
        # 批次       数据
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        # forward + backward + update
        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
        accuracy = 100 * correct / total
        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%-5d, %-5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300), 'Accuracy on train set: %.3f %%' % accuracy)
            epoch_loss = running_loss / len(train_loader)
            running_loss = 0.0
            return epoch_loss

def test():
    correct = 0
    total = 0
    # 关闭梯度计算
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
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
plt.pause(10)
