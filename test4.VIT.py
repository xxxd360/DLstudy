import torch
from functools import partial
from collections import OrderedDict
# 图像处理
import torchvision
import torchvision.transforms as transforms
# 激活函数
import torch.nn as nn
import torch.nn.functional as F
# 优化器
import torch.optim as optim
import matplotlib.pyplot as plt


transform_train = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.Resize(224),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])


trainset = torchvision.datasets.STL10(
    root='dataset_2',  download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=64, shuffle=True, num_workers=2)

testset = torchvision.datasets.STL10(
    root='dataset_2',  download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=64, shuffle=False, num_workers=2)


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # 生成与 x 的维度匹配的形状，只保持 batch 维度
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)  # 生成一个与 x 大小相同的随机张量
    random_tensor.floor_()  # 将随机张量二值化（小于 keep_prob 的值为 0，其他为 1）
    output = x.div(keep_prob) * random_tensor  # 将输入 x 缩放并与随机张量相乘，实现部分路径的丢弃
    return output  # 返回经过 drop path 操作后的张量

class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()  # 调用父类 nn.Module 的构造函数
        self.drop_prob = drop_prob  # 初始化丢弃概率

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)  # 调用上面定义的 drop_path 函数


class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, embed_dim=768, norm_layer=None):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.gird_size = (img_size[0]//patch_size[0], img_size[1]//patch_size[1])
        self.num_patches = self.gird_size[0]*self.gird_size[1]

        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H==self.img_size[0] and W==self.img_size[1], \
        f"输入图像大小{H}*{W}与模型期望大小{self.img_size[0]}*{self.img_size[1]}不匹配"
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x

class Attention(nn.Module):
    def __init__(self,
                 dim, # 输入的token维度，768
                 num_heads = 8, # 注意力的头数，为8
                 qkv_bais=False, # 生成QKV的时候是否添加偏置
                 qk_scale=None, # 用于缩放QK的系数，如果None，则使用1/sqrt(head_dim)
                 atte_drop_ration=0., # 注意力分数的dropout的比率，防止过拟合
                 proj_drop_ration=0.): # 最终投影层的dropout比例
        super().__init__()
        self.num_heads = num_heads # 注意力头数
        head_dim = dim // num_heads # 每个注意力头的维度
        self.scale = qk_scale or head_dim ** 0.5 # qk的缩放因子
        self.qkv = nn.Linear(dim, dim*3, bias=qkv_bais) # 通过全连接层生成QKV，为了并行计算，提高计算效率，参数更少
        self.att_drop = nn.Dropout(atte_drop_ration)
        self.proj_drop = nn.Dropout(proj_drop_ration)
        # 将每个head得到的输出进行concat拼接，然后通过线性变换映射回原本的嵌入dim
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape # batch, num_patchs+1, embed_dim 这个1为cls token
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C//self.num_heads).permute(2, 0, 3, 1, 4) # 方便我们之后做运算
        # 用切片拿到QKV，形状 B, self.num_heads, N, C//self.num_heads
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale    # [B, num_heads, N, N]
        attn = attn.softmax(dim=-1) # 对每行进行处理 使得每行的和为1
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x) # 防止过拟合
        return x
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        # in_features输入的维度  hidden_features 隐藏层的维度 通常为in_features的4倍，out_features维度 通常与in_features一致
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x) # 第一个全连接层
        x = self.act(x) # 激活函数
        x = self.drop(x) # 丢弃一定比例的神经元
        x = self.fc2(x)
        x = self.drop(x)
        return x
class Block(nn.Module):
    def __init__(self,
                 dim,  # 每个token的维度
                 num_heads,  # 多头自注意力的头数
                 mlp_ratio=4,  # 计算hidden_features大小 为输入的四倍
                 qkv_bais=False,
                 qk_scale=None,
                 drop_ratio=0.,  # 多头自注意力机制最后的linear后使用的dropout
                 attn_drop_ratio=0.,  # 生成qkv后的dropout
                 drop_path_ratio=0.,  # drop_path的比例
                 act_layer=nn.GELU,  # 激活函数
                 norm_layer=nn.LayerNorm):  # 正则化层
        super(Block, self).__init__()
        self.norm1 = norm_layer(dim)  # transformer encoder block中的第一个layer norm
        self.attn = Attention(dim, num_heads=num_heads, qkv_bais=qkv_bais, qk_scale=qk_scale,
                              atte_drop_ration=attn_drop_ratio, proj_drop_ration=drop_ratio)
        # 如果drop_path_ratio>0，则使用droppath，否则不做任何更改
        self.drop_path = DropPath(drop_path_ratio) if drop_path_ratio > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)  # 定义第二个layer_norm层
        mlp_hidden_dim = int(dim * mlp_ratio)  # 计算mlp第一个全连接层的节点个数
        # 定义mlp层 传入dim = mlp_hidden_dim
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop_ratio)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))  # 前向传播部分 输入的x先经过layernorm在经过multiheadattention
        x = x + self.drop_path(self.mlp(self.norm2(x)))  # 将得到的x依次通过layernorm2、mlp、drop_path
        return x

class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_c=3, num_classes=1000,
                 embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.0, qkv_bais=True,
                 qk_scale=None, representation_size=None, distilled=False, drop_ratio=0.,
                 attn_drop_ratio=0., drop_path_ratio=0., embed_layer = PatchEmbed, norm_layer=None,
                 act_layer=None):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim # embed_dim 赋值给self.embed_dim和self.num_features
        self.num_tokens = 2 if distilled else 1 # num_tokens 为1
        # 设置一个较小的参数防止除0
        norm_layer = norm_layer or partial(nn.LayerNorm, eps = 1e-6)
        act_layer = act_layer or nn.GELU()
        self.patch_embed = embed_layer(img_size=img_size, patch_size=patch_size, in_c=in_c, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches # 得到patches的个数
        # 使用nn.Parameter构建可训练的参数，用零矩阵初始化，第一个为batch维度
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # pos_embed 大小与concat拼接后的大小一致 197, 768
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim))
        self.pos_drop = nn.Dropout(p = drop_ratio)
        # 根据传入的drop_path_ratio构建等差数列从0到drop_path_ratio，有depth个元素
        dpr = [x.item() for x in torch.linspace(0, drop_path_ratio, depth)]
        # 使用nn.Sequential将列表中的所有模块打包为一个整体
        self.block = nn.Sequential(*[
            Block(dim = embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bais=qkv_bais, qk_scale=qk_scale,
                  drop_ratio=drop_ratio, attn_drop_ratio=attn_drop_ratio, drop_path_ratio=dpr[i],
                  norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim) # 通过transformer后的layernorm

        if representation_size and not distilled:
            self.has_logits = True
            self.pre_logits = nn.Sequential(OrderedDict([
                ("fc", nn.Linear(embed_dim, representation_size)),
                ("act", nn.Tanh())
            ]))
        else:
            self.has_logits = False
            self.pre_logits = nn.Identity() # pre_logits不做任何处理
        # 分类头
        self.head = nn.Linear(self.num_features, num_classes) if num_classes>0 else nn.Identity()
        self.head_dist = None
        if distilled:
            self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if num_classes>0 else nn.Identity()

        # 权重初始化
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        if self.dist_token is not None:
            nn.init.trunc_normal_(self.dist_token, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(_init_vit_weights)

    def forward_features(self, x):
        # B C H W -> B num_patches embed_dim
        x = self.patch_embed(x)
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        # 如果dist_token存在则拼接dist_token和cls_token，否则只拼接cls_token和输入的patch特征x
        if self.dist_token is None:
            x = torch.cat((cls_token, x), dim=1) # B 197 768 在维度1上面拼接
        else:
            x = torch.cat((cls_token, self.dist_token.expand(x.shape[0], -1, -1), x), dim=1)
        x = self.pos_drop(x + self.pos_embed)
        x = self.block(x)
        x = self.norm(x)
        if self.dist_token is None: # dist_token为None 提取cls_token对应的输出
            return self.pre_logits(x[:, 0])
        else:
            return x[:, 0], x[:, 1]

    def forward(self, x):
        x = self.forward_features(x)
        if self.head_dist is not None:
            # 分别通过head和head-dist进行预测
            x, x_dist = self.head(x[0]), self.head_dist(x[1])
            # 如果是训练模式且不是脚本模式
            if self.training and not torch.jit.is_scripting():
                # 则返回两个头部的预测结果
                return x, x_dist
        else:
            x = self.head(x) # 最后的linear 全连接层
        return x

def _init_vit_weights(m):
    # 判断模块m是否是nn.Linear
    if isinstance(m, nn.Linear):
        nn.init.trunc_normal_(m.weight, std=.01)
        if m.bias is not None: # 如果线性层存在偏置项
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out") # 对卷积层的权重做一个初始化 适用于卷积
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m, nn.Conv2d):
        nn.init.kaiming_normal_(m.weight, mode="fan_out")  # 对卷积层的权重做一个初始化 适用于卷积
        if m.bias is not None:
            nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)  # 对层归一化的权重初始化为1
def vit_base_patch16_224(num_classes: int = 1000, pretrained=False):
    model = VisionTransformer(img_size=224,
                              patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=8,
                              representation_size=None,
                              num_classes=num_classes)
    return model



model = vit_base_patch16_224(num_classes=10)
# 损失函数与优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 训练函数
def train(epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

        if batch_idx % 50 == 0:
            print(f'Train Epoch: {epoch} | Batch: {batch_idx} | Loss: {running_loss/(batch_idx+1):.3f} | '
                  f'Acc: {100.*correct/total:.3f}% ({correct}/{total})')

# 测试函数
def test(epoch):
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    print(f'\nTest Epoch: {epoch} | Loss: {test_loss/len(testloader):.3f} | '
          f'Acc: {100.*correct/total:.3f}% ({correct}/{total})\n')
    return 100.*correct/total

# 执行训练与测试
if __name__ == '__main__':
    epochs = 50
    best_acc = 0
    for epoch in range(epochs):
        train(epoch)
        acc = test(epoch)
       # 保存最佳模型
        if acc > best_acc:
           best_acc = acc
           torch.save(model.state_dict(), 'vit_best.pth')
    print(f'Best Test Accuracy: {best_acc:.3f}%')

