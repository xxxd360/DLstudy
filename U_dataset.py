import os
from PIL import Image
import numpy as np
from torch.utils.data import Dataset


class DriveDataset(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset, self).__init__()
        # 判断是训练集还是测试集，设置相应的标志
        self.flag = "training" if train else "test"
        # 生成数据集路径，data_root是训练集或测试集的根目录
        data_root = os.path.join(root, "DRIVE", self.flag)
        # 检查路径是否存在，如果不存在就报错
        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms
        # 获取所有.tif格式的图像文件名
        img_names = [i for i in os.listdir(os.path.join(data_root, "images")) if i.endswith(".tif")]
        # 生成图像路径的列表
        self.img_list = [os.path.join(data_root, "images", i) for i in img_names]
        # 为每个图像生成对应的人工标注路径
        self.manual = [os.path.join(data_root, "1st_manual", i.split("_")[0] + "_manual1.gif")
                       for i in img_names]
        # 检查每个人工标注文件是否存在
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        # 为每个图像生成对应的ROI掩码路径
        self.roi_mask = [os.path.join(data_root, "mask", i.split("_")[0] + f"_{self.flag}_mask.gif")
                         for i in img_names]
        # 检查每个ROI掩码文件是否存在
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

    def __getitem__(self, idx):
        """
        这个方法返回数据集中某个特定位置（idx）的图像和对应的掩码。
        1. 读取图像、人工标注图像和ROI掩码图像。
        2. 将人工标注转化为[0, 1]范围的数值，ROI掩码会反转（因为它可能是黑白反转的）。
        3. 合并人工标注和ROI掩码，得到最终的掩码。
        4. 如果有转换函数（transforms），就对图像和掩码一起进行处理。
        """
        # 打开原始图像并转换为RGB格式
        img = Image.open(self.img_list[idx]).convert('RGB')
        # 打开人工标注图像并转换为灰度模式
        manual = Image.open(self.manual[idx]).convert('L')
        # 将人工标注的值转换为[0, 1]之间
        manual = np.array(manual) / 255
        # 打开ROI掩码图像并转换为灰度模式
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        # 反转ROI掩码图像，将黑色区域变成白色区域，白色区域变成黑色
        roi_mask = 255 - np.array(roi_mask)
        # 合并人工标注和ROI掩码，得到最终的掩码，超出[0, 255]范围的部分会被裁剪
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 将NumPy格式的掩码转换回PIL格式，因为转换函数（transforms）通常是针对PIL格式的
        mask = Image.fromarray(mask)

        # 如果提供了转换函数，就应用它们（例如，数据增强等）
        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        # 返回处理过的图像和掩码
        return img, mask

    def __len__(self):
        # 返回数据集中的样本数量，即图像的数量
        return len(self.img_list)

    @staticmethod
    def collate_fn(batch):
        """
        这个函数用于将多个样本合并成一个批次（batch）。
        - batch：是由多个图像和目标掩码组成的列表
        - 使用`cat_list`函数将图像和掩码拼接成一个统一大小的批次
        """
        # 将批次中的所有图像和目标掩码分别取出
        images, targets = list(zip(*batch))
        # 将图像列表按最大尺寸拼接成一个批次
        batched_imgs = cat_list(images, fill_value=0)
        # 将掩码列表按最大尺寸拼接成一个批次
        batched_targets = cat_list(targets, fill_value=255)
        # 返回拼接好的图像和掩码
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    """
    这个函数将多个图像拼接成一个批次。
    所有图像会被填充成相同的尺寸，填充部分用`fill_value`来填充。
    - images：要拼接的图像列表
    - fill_value：填充区域的值，默认是0（黑色）
    """
    # 找到所有图像中最大的尺寸（宽和高）
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    # 创建一个形状为(batch_size, max_height, max_width)的零矩阵，用来存放批次中的图像
    batch_shape = (len(images),) + max_size
    # 创建一个全是`fill_value`的矩阵，作为初始的空白批次
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    # 将每个图像复制到对应的位置，保证原图大小不变，超出部分用填充值填充
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    # 返回拼接好的批次图像
    return batched_imgs