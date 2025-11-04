import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """(卷积 -> BN -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """下采样：MaxPool -> DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """上采样：反卷积/双线性插值 + 拼接 + DoubleConv"""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        # 双线性插值上采样（轻量）或反卷积（参数化）
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels)
        else:
            # 修复反卷积通道数：输入通道=in_channels，输出通道=out_channels
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)  # in_channels = out_channels（上采样） + out_channels（编码器特征）

    def forward(self, x1, x2):
#        import pdb; pdb.set_trace()
        # 上采样并调整尺寸
        x1 = self.up(x1)
        # 计算拼接所需的偏移（处理尺寸不匹配）
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # 拼接编码器特征(x2)和解码器特征(x1)
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    """输出层：1x1卷积映射到类别数"""
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    """U-Net主结构"""
    def __init__(self, n_channels=3, n_classes=1, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels  # 输入通道（如RGB=3，灰度图=1）
        self.n_classes = n_classes    # 输出类别（如二分类=1，多分类=N）
        self.bilinear = bilinear      # 是否用双线性插值上采样

        # 编码器
        self.inc = DoubleConv(n_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if bilinear else 1  # 双线性插值时通道数减半
        self.down4 = Down(512, 1024 // factor)

        # 解码器
        self.up1 = Up(1024, 512 // factor, bilinear)
        self.up2 = Up(512, 256 // factor, bilinear)
        self.up3 = Up(256, 128 // factor, bilinear)
        self.up4 = Up(128, 64, bilinear)

        # 输出层
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
#        import pdb; pdb.set_trace()
        # 编码器下采样，保存中间特征用于跳跃连接
        x1 = self.inc(x)       # 输入 -> 64通道
        x2 = self.down1(x1)    # 下采样1 -> 128通道
        x3 = self.down2(x2)    # 下采样2 -> 256通道
        x4 = self.down3(x3)    # 下采样3 -> 512通道
        x5 = self.down4(x4)    # 下采样4 -> 1024通道（瓶颈层）

        print('======== encoder')
        print("x1 : ", x1.shape)
        print("x2 : ", x2.shape)
        print("x3 : ", x3.shape)
        print("x4 : ", x4.shape)
        print("x5 : ", x5.shape)
        
        print('======== decoder')
        # 解码器上采样，拼接编码器特征
        print(x5.shape)
        x = self.up1(x5, x4)   # 上采样+拼接x4 -> 512通道
        print(x.shape)
        x = self.up2(x, x3)    # 上采样+拼接x3 -> 256通道
        print(x.shape)
        x = self.up3(x, x2)    # 上采样+拼接x2 -> 128通道
        print(x.shape)
        x = self.up4(x, x1)    # 上采样+拼接x1 -> 64通道
        print(x.shape)

        logits = self.outc(x)  # 输出类别概率图
        return logits


# 测试代码
if __name__ == "__main__":
    # 实例化模型（输入3通道RGB图，输出2类分割）
    model = UNet(n_channels=3, n_classes=2, bilinear=False)
    # 随机生成输入（batch_size=2, 3通道, 256x256尺寸）
    x = torch.randn(2, 3, 256, 256)
    # 前向传播
    output = model(x)
    # 输出形状：(batch_size, 类别数, 高度, 宽度)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")  # 应输出 (2, 2, 256, 256)
