# 任务一
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import torch
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

# ===================== 1. 张量基本操作 =====================
# 创建2x3张量
x = torch.tensor([[1, 2, 3], [4, 5, 6]])
y = torch.tensor([[7, 8, 9], [10, 11, 12]])
print("创建的张量x:\n", x)

# 张量加法
add_result = x + y
print("张量加法结果:\n", add_result)

# 元素级乘法
mul_result = x * y
print("张量元素级乘法结果:\n", mul_result)

# 矩阵乘法（x转置后与y相乘）
matmul_result = torch.matmul(x.T, y)
print("张量矩阵乘法结果:\n", matmul_result)


# ===================== 2. 自动求导(autograd)使用 =====================
x = torch.tensor(2.0, requires_grad=True)
y = x ** 3 + 3 * x + 1
y.backward()
print("x的梯度dy/dx:", x.grad)


# ===================== 3. 加载并显示MNIST样本 =====================
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root='dataset_1', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4, shuffle=True)
# 构建迭代器
data_iter = iter(train_loader)
images, labels = next(data_iter)

fig, axes = plt.subplots(1, 4, figsize=(12, 3))
for i in range(4):
    axes[i].imshow(images[i].squeeze().numpy(), cmap='gray')
    axes[i].set_title(f"Label: {labels[i].item()}")
plt.show()