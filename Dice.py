import torch
import torch.nn as nn


def build_target(target: torch.Tensor, num_classes: int = 2, ignore_index: int = -100):
    """构建Dice系数需要的目标标签"""
    dice_target = target.clone()  # 克隆目标标签，避免直接修改原数据
    if ignore_index >= 0:
        # 如果存在ignore_index，找到目标标签中所有等于ignore_index的部分
        ignore_mask = torch.eq(target, ignore_index)
        dice_target[ignore_mask] = 0  # 将ignore_index部分设置为0
        # 将目标标签进行one-hot编码，转换为[N, H, W, C]的格式
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()
        dice_target[ignore_mask] = ignore_index  # 恢复ignore_index的区域
    else:
        # 如果没有ignore_index，直接进行one-hot编码
        dice_target = nn.functional.one_hot(dice_target, num_classes).float()

    # 调整维度顺序，从[N, H, W, C]变为[N, C, H, W]
    return dice_target.permute(0, 3, 1, 2)


def dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """计算一个batch中所有图片某个类别的Dice系数"""
    d = 0.  # 初始化Dice系数
    batch_size = x.shape[0]  # 获取batch的大小
    for i in range(batch_size):
        # 将第i张图像和目标标签展平为一维数组
        x_i = x[i].reshape(-1)
        t_i = target[i].reshape(-1)

        if ignore_index >= 0:
            # 如果有ignore_index，找到目标标签中不为ignore_index的区域
            roi_mask = torch.ne(t_i, ignore_index)
            x_i = x_i[roi_mask]  # 只保留不为ignore_index的部分
            t_i = t_i[roi_mask]  # 目标标签也做同样的处理

        # 计算交集（预测值和目标值的点积）
        inter = torch.dot(x_i, t_i)
        # 计算并集（预测区域总和 + 目标区域总和）
        sets_sum = torch.sum(x_i) + torch.sum(t_i)

        if sets_sum == 0:
            # 如果并集为0（预测和目标都没有预测到任何目标），就直接返回2 * 交集
            sets_sum = 2 * inter

        # 计算Dice系数，并且避免除0错误，加入一个小常数epsilon
        d += (2 * inter + epsilon) / (sets_sum + epsilon)

    # 返回batch内所有图片的平均Dice系数
    return d / batch_size


def multiclass_dice_coeff(x: torch.Tensor, target: torch.Tensor, ignore_index: int = -100, epsilon=1e-6):
    """计算所有类别的Dice系数平均值"""
    dice = 0.  # 初始化Dice系数
    for channel in range(x.shape[1]):  # 对每一个类别（通道）计算Dice系数
        dice += dice_coeff(x[:, channel, ...], target[:, channel, ...], ignore_index, epsilon)

    # 返回所有类别的平均Dice系数
    return dice / x.shape[1]

def dice_loss(x: torch.Tensor, target: torch.Tensor, multiclass: bool = False, ignore_index: int = -100):
    """计算Dice损失（目标是最小化该损失）"""
    x = nn.functional.softmax(x, dim=1)  # 对模型输出进行softmax，转化为概率分布
    fn = multiclass_dice_coeff if multiclass else dice_coeff  # 根据是否多分类选择对应的函数
    # 计算Dice损失，目标是最小化，所以是1减去Dice系数
    return 1 - fn(x, target, ignore_index=ignore_index)