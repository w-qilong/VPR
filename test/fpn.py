import torch
import torch.nn as nn
import torch.nn.functional as F

class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        # 横向连接层
        self.lateral_convs = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=1) for in_channels in in_channels_list]
        )
        # 自顶向下的卷积层
        self.fpn_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1) for _ in in_channels_list]
        )

    def forward(self, inputs):
        # 自顶向下的路径
        last_inner = self.lateral_convs[-1](inputs[-1])
        results = [self.fpn_convs[-1](last_inner)]
        for feature, lateral_conv, fpn_conv in zip(
            reversed(inputs[:-1]), reversed(self.lateral_convs[:-1]), reversed(self.fpn_convs[:-1])
        ):
            lateral_feature = lateral_conv(feature)
            last_inner = F.interpolate(last_inner, size=lateral_feature.shape[-2:], mode='nearest') + lateral_feature
            results.insert(0, fpn_conv(last_inner))
        
        # 将所有特征图下采样到目标分辨率（输入图像的一半）
        target_resolution = (inputs[0].shape[-2] // 2, inputs[0].shape[-1] // 2)
        results = [F.interpolate(result, size=target_resolution, mode='nearest') for result in results]
        
        # 将所有特征图在通道维度上进行拼接
        fused_feature = torch.cat(results, dim=1)
        
        return fused_feature

# 示例用法
if __name__ == "__main__":
    # 假设VGG的输出特征图通道数为[64, 128, 256, 512]
    fpn = FPN(in_channels_list=[128, 256, 512], out_channels=128)
    
    # 假设输入特征图的尺寸
    inputs = [
        torch.rand(1, 128, 56, 56),
        torch.rand(1, 256, 28, 28),
        torch.rand(1, 512, 14, 14),
    ]
    
    # 获取FPN输出
    fused_feature = fpn(inputs)
    print(f"融合后的特征图尺寸: {fused_feature.shape}")