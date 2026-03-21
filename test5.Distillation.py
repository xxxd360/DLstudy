import torch
from torch import nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchinfo import summary
from tqdm import tqdm

# torch.manual_seed(0)
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.benchmark = True

# 载入训练集
train_dataset = torchvision.datasets.MNIST(
    root="dataset_1",
    train=True,
    transform=transforms.ToTensor(),
    download=True
)
# 载入测试集
test_dataset = torchvision.datasets.MNIST(
    root="dataset_1",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)
# 生成DataLoader
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=False)

class TeacherModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(TeacherModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x


model = TeacherModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 6

for epoch in range(epochs):
    model.train()
    for data, targets in tqdm(train_loader):

        preds = model(data)
        loss = criterion(preds, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    acc = (num_correct / num_samples).item()
    model.train()
    print(f'Epoch:{epoch + 1}\t Accuracy:{acc:.4f}')

teacher_model = model

class StudentModel(nn.Module):
    def __init__(self, in_channels=1, num_classes=10):
        super(StudentModel, self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, num_classes)

    def forward(self, x):
        x = x.view(-1, 784)
        x = self.fc1(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.relu(x)

        x = self.fc3(x)
        return x


teacher_model.eval()
model = StudentModel()
model.train()
temp = 7

hard_loss = nn.CrossEntropyLoss()
alpha = 0.3
soft_loss = nn.KLDivLoss(reduction="batchmean")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
epochs = 3

for epoch in range(epochs):
    for data, targets in tqdm(train_loader):
        data = data
        targets = targets

        with torch.no_grad():
            teacher_preds = teacher_model(data)

        student_preds = model(data)
        student_loss = hard_loss(student_preds, targets)

        ditillation_loss = soft_loss(
            F.softmax(student_preds / temp, dim=1),
            F.softmax(teacher_preds / temp, dim=1)
        )

        loss = alpha * student_loss + (1 - alpha) * ditillation_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        for x, y in test_loader:
            preds = model(x)
            predictions = preds.max(1).indices
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)
    acc = (num_correct / num_samples).item()
    model.train()
    print(f'Epoch:{epoch + 1}\t Accuracy:{acc:.4f}')

