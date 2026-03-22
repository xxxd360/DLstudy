import torch.optim as optim
from DLstudy.U_dataset import DriveDataset
from Unet import UNet
import torch
import torch.utils as utils
from Dice import dice_loss
model = UNet(n_channels=3, n_classes=2)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
criterion = dice_loss.DiceLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()  # 设置模型为训练模式
    metric_logger = utils.MetricLogger(delimiter="  ")  # 用于记录日志
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))  # 记录学习率
    header = 'Epoch: [{}]'.format(epoch)

    # 如果是二分类，设置交叉熵中的损失权重（背景和前景的权重不同）
    if num_classes == 2:
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)  # 背景权重1.0，前景权重2.0
    else:
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        # 使用自动混合精度（如果有的话）进行前向传播
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)  # 前向传播
            # 计算损失
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)

        optimizer.zero_grad()  # 清空梯度
        if scaler is not None:
            # 使用混合精度训练时，进行反向传播和参数更新
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规反向传播
            loss.backward()
            optimizer.step()

        lr_scheduler.step()  # 更新学习率

        # 获取当前学习率
        lr = optimizer.param_groups[0]["lr"]
        # 更新日志
        metric_logger.update(loss=loss.item(), lr=lr)

    return metric_logger.meters["loss"].global_avg, lr  # 返回平均损失和当前学习率
if __name__ == '__main__':
    train_dataset = DriveDataset(root="DRIVE", train=True)