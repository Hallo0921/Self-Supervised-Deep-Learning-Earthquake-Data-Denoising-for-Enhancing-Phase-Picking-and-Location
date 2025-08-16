import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F


class ChannelWiseFusion(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        # 分量注意力机制
        self.component_att = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // 3, 1),
            nn.Softplus(),
            nn.Conv2d(in_channels // 3, in_channels, 1),
            nn.Sigmoid()
        )
        self.channel_interact = nn.Conv2d(in_channels, in_channels, 3,
                                          padding=1, groups=3)

    def forward(self, x):
        component_weight = self.component_att(x)
        return x * component_weight + self.channel_interact(x)


class TriComponentProcessor(nn.Module):
    def __init__(self, components=3):
        super().__init__()
        self.components = components

        # 编码器结构 (每个分量独立处理)
        self.enc1 = nn.Sequential(
            nn.Conv2d(2*components, 24, 3, padding=1, groups=3),
            ChannelWiseFusion(24),
            nn.MaxPool2d(2),
            nn.Softplus()
        )

        # 中间特征处理器
        self.enc2 = nn.Sequential(
            nn.Conv2d(24, 48, 3, padding=1, groups=3),
            nn.BatchNorm2d(48),
            nn.MaxPool2d(2),
            nn.Softplus()
        )

        self.enc3 = nn.Sequential(
            nn.Conv2d(48, 72, 3, padding=1, groups=3),
            nn.BatchNorm2d(72),
            nn.MaxPool2d(2),
            nn.Softplus()
        )

        # 解码器结构
        self.dec1 = nn.Sequential(
            nn.ConvTranspose2d(72, 48, 3, stride=2,
                               padding=1, output_padding=1),
            ChannelWiseFusion(48),
            nn.Softplus()
        )
        self.dec2 = nn.Sequential(
            nn.ConvTranspose2d(48, 24, 3, stride=2,
                               padding=1, output_padding=1),
            ChannelWiseFusion(24),
            nn.Softplus()
        )
        self.dec3 = nn.Sequential(
            nn.ConvTranspose2d(24, components, 3, stride=2,
                               padding=1, output_padding=1),
            ChannelWiseFusion(3),
            nn.Softplus()
        )

    def forward(self, real_imag):
        outputs = []
        x = self.enc1(real_imag)
        outputs.append(x)
        x = self.enc2(x)
        outputs.append(x)
        x = self.enc3(x)
        outputs.append(x)
        x = self.dec1(x)
        outputs.append(x)
        x = self.dec2(x)
        outputs.append(x)
        x = self.dec3(x)
        outputs.append(x)
        # 动态调整输出尺寸
        if x.size()[-2:] != real_imag.size()[-2:]:
            x = F.interpolate(x, size=real_imag.shape[-2:],
                              mode='bilinear', align_corners=True)
        return x, outputs
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == '__main__':
    # x1 = torch.randn(128, 6, 11, 601)
    # net = TriComponentProcessor()
    # print(net(x1).shape)
    # print(f"Total trainable parameters: {count_parameters(net)}")
    x1 = torch.randn(128, 6, 11, 601)
    net = TriComponentProcessor()
    output, layer_outputs = net(x1)  # 获取最终输出和各层输出列表
    print(output.shape)

    print(f"Total trainable parameters: {count_parameters(net)}")

    # 打印各层输出的形状
    for idx, out in enumerate(layer_outputs):
        print(f"Layer {idx + 1} output shape: {out.shape}")