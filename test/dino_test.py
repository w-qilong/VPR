import torch

# model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14').cuda()

# img = torch.randn(1, 3, 224, 224).cuda()
# img=model.prepare_tokens_with_masks(img)
# print(img.shape)


import torch
import torch.nn as nn

# 定义反卷积层
deconv = nn.ConvTranspose2d(
    in_channels=1,  # 输入通道数
    out_channels=1,  # 输出通道数
    kernel_size=7,  # 卷积核大小
    stride=7,  # 步长
    padding=0 # 填充
)

# 创建一个16x16的特征图
input_feature_map = torch.randn(1, 1, 16, 16)  # (batch_size, channels, height, width)

# 使用反卷积进行上采样
output_feature_map = deconv(input_feature_map)

print("输出特征图的大小:", output_feature_map.shape)


import torch
import torch.nn as nn
import torchvision.models as models

class VGG19FeatureExtractor(nn.Module):
    def __init__(self):
        super(VGG19FeatureExtractor, self).__init__()
        # 加载预训练的 VGG19 模型
        vgg19 = models.vgg19(pretrained=False).features
        
        # 提取特定层的特征
        self.stage1 = nn.Sequential(*vgg19[:9])   # 1/2 分辨率
        self.stage2 = nn.Sequential(*vgg19[9:18]) # 1/4 分辨率
        self.stage3 = nn.Sequential(*vgg19[18:27])# 1/8 分辨率

    def forward(self, x):
        # 提取多尺度特征图
        feature1 = self.stage1(x)  # 1/2 分辨率
        feature2 = self.stage2(feature1)  # 1/4 分辨率
        feature3 = self.stage3(feature2)  # 1/8 分辨率
        
        return feature1, feature2, feature3

# 示例用法
if __name__ == "__main__":
    # 创建特征提取器实例
    feature_extractor = VGG19FeatureExtractor()
    
    # 假设输入图像尺寸为 (224, 224)
    input_image = torch.rand(1, 3, 224, 224)
    
    # 获取多尺度特征图
    feature1, feature2, feature3 = feature_extractor(input_image)
    print(f"1/2 分辨率特征图尺寸: {feature1.shape}")
    print(f"1/4 分辨率特征图尺寸: {feature2.shape}")
    print(f"1/8 分辨率特征图尺寸: {feature3.shape}")