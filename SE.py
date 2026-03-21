import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.datasets import DatasetFolder,ImageFolder
import torch.nn.functional as F
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import matplotlib.pyplot as plt
batch_size = 64
num_epochs=10
num_workers = 4
transform = transforms.Compose([transforms.ToTensor(),transforms.Normalize([0.1307],[0.3081])])
dataset = ImageFolder(root = r"D:\pytorch.pycharm\RM\armor_8c_new",
                      transform = transform)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_loader = DataLoader( dataset=  train_dataset ,
                           shuffle = True,
                           batch_size=batch_size,
                           num_workers=num_workers,
                           pin_memory=True)
val_loader = DataLoader(dataset=val_dataset,
                        shuffle =False,
                        batch_size=batch_size)
#这里使用SE_ResidualBlock
class ResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResidualBlock, self).__init__()
        # 这里需根据实际残差块结构补充，示例中假设包含卷积、激活等操作
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.relu(out)
        out = self.conv2(out)
        out += residual
        out = self.relu(out)
        return out

class SEblock(nn.Module):
    def __init__(self,channel):
        super(SEblock,self).__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channel, channel // 16, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 16, channel, kernel_size=1, bias=False),
            nn.Sigmoid()
        )
    def forward(self,x):
        wight = self.fc(x)
        return x*wight



class Net(nn.Module):
     def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, kernel_size=5)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5)
        self.mp = nn.MaxPool2d(2)
        self.rbblock1 = ResidualBlock(16)
        self.rbblock2 = ResidualBlock(32)
        self.SEblock1 = SEblock(16)
        self.SEblock2 = SEblock(32)
        self.fc = nn.Linear(5408, 8)

     def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.conv1(x)))
        x = self.rbblock1(x)
        x = self.SEblock1(x)
        x = self.mp(F.relu(self.conv2(x)))
        x = self.rbblock2(x)
        x = self.SEblock2(x)
        x = x.view(in_size, -1)
        x = self.fc(x)
        return x

model = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)
def train(epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        inputs, target = inputs.to(device), target.to(device)
        optimizer.zero_grad()


        outputs = model(inputs)
        loss = criterion(outputs,target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, dim=1)
        total += target.size(0)
        correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print('[%-5d, %-5d] loss: %.3f' % (num_epochs + 1, len(train_loader), running_loss / len(train_loader)),
          'Accuracy on batch: %.3f %%' % accuracy)
    epoch_loss = running_loss / len(train_loader)
    return epoch_loss,accuracy

def val():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            inputs, target = data
            inputs, target = inputs.to(device), target.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, dim=1)
            total += target.size(0)
            correct += (predicted == target).sum().item()
    accuracy = 100 * correct / total
    print(f'Accuracy:{accuracy:.5f}')
    return accuracy


if __name__ == "__main__":
    accuracy_list = []
    loss_list= []
    save_dir = './checkpoints'
    os.makedirs(save_dir, exist_ok=True)
    best_loss = float('inf')
    for epoch in range(num_epochs):
        epoch_loss,accuracy = train(epoch)
        loss_list.append(epoch_loss)
        accuracy_list.append(accuracy)
        torch.save(model.state_dict(), f'{save_dir}/latest_model.pth')

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')

        if epoch % 10 == 9:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
            }, f'{save_dir}/checkpoint_epoch_{epoch}.pth')
    val()
    fig = plt.figure(figsize=(8, 6))
    ax1 = fig.add_subplot(211)
    ax2 = fig.add_subplot(212)
    ax1.plot(range(num_epochs), loss_list)
    ax2.plot(range(num_epochs),accuracy_list)
    ax1.set_title('Loss per epoch')
    ax1.set_ylabel('Loss')
    ax1.set_xlabel('Epoch')
    ax2.set_title('Accuracy per epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_xlabel('Epoch')
    plt.show()
