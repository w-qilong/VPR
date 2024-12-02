import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tvm
from prettytable import PrettyTable
import torchvision.models as models
from dinov2.hub.backbones import (
    dinov2_vitb14,
    dinov2_vits14,
    dinov2_vitl14,
    dinov2_vitg14,
)


class Dinov2CoarseFeatureExtractor(nn.Module):
    def __init__(self, backbone_size="small", topk=32):
        super().__init__()
        self.backbone, self.checkpoint_path, self.patch_dim = self._initialize_backbone(
            backbone_size
        )
        self.topk = topk
        self._load_checkpoint(self.checkpoint_path)
        self._freeze_non_adapter_params()
        self._print_param_stats()

    def _initialize_backbone(self, backbone_size):
        # 根据backbone_size选择合适的backbone，并返回其对应的检查点路径
        backbone_map = {
            "small": dinov2_vits14,
            "base": dinov2_vitb14,
            "large": dinov2_vitl14,
            "giant": dinov2_vitg14,
        }
        patch_dim = {"s": 384, "b": 768, "l": 1024, "g": 1024}
        checkpoint_path = f"/home/cartolab3/.cache/torch/hub/checkpoints/dinov2_vit{backbone_size[0]}14_pretrain.pth"
        return (
            backbone_map[backbone_size](pretrained=False),
            checkpoint_path,
            patch_dim[backbone_size[0]],
        )

    def _load_checkpoint(self, checkpoint_path):
        # 加载模型的检查点
        with open(self.checkpoint_path, "rb") as f:
            checkpoint = torch.load(f)
        backbone_dict = self.backbone.state_dict()
        backbone_dict.update(checkpoint.items())
        self.backbone.load_state_dict(backbone_dict)

    def _freeze_non_adapter_params(self):
        # 冻结backbone中不属于adapter的参数
        for name, param in self.backbone.named_parameters():
            if "adapter" not in name:
                param.requires_grad = False

    def _print_param_stats(self):
        # 打印模型参数统计信息
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        table = PrettyTable()
        table.field_names = ["统计项", "参数值"]
        table.add_row(["总参数量", f"{total_params:,} ({total_params/1e6:.2f}M)"])
        table.add_row(
            ["可训练参数量", f"{trainable_params:,} ({trainable_params/1e6:.2f}M)"]
        )
        table.add_row(["可训练参数占比", f"{(trainable_params/total_params)*100:.2f}%"])

        print("\n模型参数统计:")
        print(table)

    def forward(self, x):
        # 前向传播，获取cls_token和patch_tokens
        coarse_features = self.backbone.forward_features(x)
        cls_token = coarse_features["x_norm_clstoken"]
        patch_tokens = coarse_features["x_norm_patchtokens"]

        # 计算cls_token与patch_tokens之间的相关性
        # similarity = torch.einsum("bd,bnd->bn", cls_token, patch_tokens)

        # 获取与cls_token最相关的top k个patch的index
        # _, topk_indices = torch.topk(similarity, self.topk, dim=1)

        # 扩展索引维度以匹配特征维度
        # expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, patch_tokens.shape[-1])
        # 使用gather选择对应的patch_tokens
        # selected_patches = torch.gather(patch_tokens, 1, expanded_indices)

        return cls_token, patch_tokens





# 用于提取pyramid特征图的VGG19模型,并使用FPN进行特征图融合
"""1/2 分辨率特征图尺寸: torch.Size([1, 128, 112, 112])
1/4 分辨率特征图尺寸: torch.Size([1, 256, 56, 56])
1/8 分辨率特征图尺寸: torch.Size([1, 512, 28, 28])"""


class VGG19FeatureExtractor(nn.Module):
    def __init__(self, pretrained=True, fine_fused_dim=128):
        super(VGG19FeatureExtractor, self).__init__()
        # 加载预训练的 VGG19 模型
        vgg19 = models.vgg19(pretrained=pretrained).features

        # 初始化FPN
        self.fpn = FPN(in_channels_list=[128, 256, 512], out_channels=fine_fused_dim)

        # 提取特定层的特征
        self.stage1 = nn.Sequential(*vgg19[:9])  # 1/2 分辨率
        self.stage2 = nn.Sequential(*vgg19[9:18])  # 1/4 分辨率
        self.stage3 = nn.Sequential(*vgg19[18:27])  # 1/8 分辨率

    def forward(self, x):
        # 提取多尺度特征图
        feature1 = self.stage1(x)  # 1/2 分辨率
        feature2 = self.stage2(feature1)  # 1/4 分辨率
        feature3 = self.stage3(feature2)  # 1/8 分辨率

        # 使用FPN进行特征图融合
        fine_fused_features = self.fpn([feature1, feature2, feature3])

        return fine_fused_features


# 特征图融合网络FPN
class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        super(FPN, self).__init__()
        # 横向连接层
        self.lateral_convs = nn.ModuleList(
            [
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
                for in_channels in in_channels_list
            ]
        )
        # 自顶向下的卷积层
        self.fpn_convs = nn.ModuleList(
            [
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
                for _ in in_channels_list
            ]
        )

    def forward(self, inputs):
        # 自顶向下的路径
        last_inner = self.lateral_convs[-1](inputs[-1])
        results = [self.fpn_convs[-1](last_inner)]
        for feature, lateral_conv, fpn_conv in zip(
            reversed(inputs[:-1]),
            reversed(self.lateral_convs[:-1]),
            reversed(self.fpn_convs[:-1]),
        ):
            lateral_feature = lateral_conv(feature)
            last_inner = (
                F.interpolate(
                    last_inner, size=lateral_feature.shape[-2:], mode="nearest"
                )
                + lateral_feature
            )
            results.insert(0, fpn_conv(last_inner))

        # 将所有特征图下采样到目标分辨率（输入图像的一半）
        results = [
            F.interpolate(result, size=inputs[2].shape[-2], mode="nearest")
            for result in results
        ]

        # 将所有特征图在通道维度上进行拼接
        fused_feature = torch.cat(results, dim=1)

        return fused_feature


# DINOV2的patch size为14，提取特征图的固定分辨率为14*14，不适用于利用细粒度局部特征匹配的重排序过程
# 因此，使用CNN网络VGG19提取特征图，并将其与DINOV2的patch_tokens拼接，作为细粒度局部特征匹配的输入，
# 通过引入全局语义信息，提升细粒度局部特征匹配的性能
class Dinov2Adapter(nn.Module):
    def __init__(self, backbone_size="small", vgg_pretrained=True, fine_fused_dim=128, topk=32):
        super().__init__()
        self.coarse_feature_extractor = Dinov2CoarseFeatureExtractor(backbone_size, topk)
        self.fine_feature_extractor = VGG19FeatureExtractor(
            pretrained=vgg_pretrained, fine_fused_dim=fine_fused_dim
        )
        self.patch_dim = self.coarse_feature_extractor.patch_dim
        self.fine_fused_dim = fine_fused_dim

        # 添加线性层用于降维
        self.reduce_dim = nn.Linear(384 + self.patch_dim, fine_fused_dim)

        # 选择由DINOV2产生的与CLS token最相关的topk个patch_tokens进行互最近邻匹配
        self.topk = topk

        # 打印模型参数总量，以及可训练参数数量，单位为M
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"Total parameters: {total_params / 1e6}M")
        print(f"Trainable parameters: {trainable_params / 1e6}M")

    def forward(self, x):
        # 获取输入图像的尺寸
        B, C, H, W = x.shape

        # 使用DINOV2提取cls_token和patch_tokens
        cls_token, patch_tokens = self.coarse_feature_extractor(x)
        patch_tokens = patch_tokens.permute(0, 2, 1).reshape(
            B, self.patch_dim, H // 14, W // 14
        )

        # 使用VGG19提取特征图,并将其融合为多尺度特征图
        fine_fused_features = self.fine_feature_extractor(x)

        # 将patch_tokens重采样到与fine_fused_features相同的尺寸，并进行拼接
        patch_tokens = F.interpolate(
            patch_tokens, size=fine_fused_features.shape[-2:], mode="nearest"
        )
        fused_features = torch.cat([patch_tokens, fine_fused_features], dim=1)

        # 降维
        B, C, H, W = fused_features.shape
        fused_features = fused_features.view(B, C, -1).permute(0, 2, 1)  # (B, H*W, C)
        reduced_features = self.reduce_dim(fused_features)  # (B, H*W, 128)
        reduced_features = reduced_features.permute(0, 2, 1).view(
            B, -1, H, W
        )  # (B, 128, H, W)

        # 标准化
        normalized_CNN_DINO_fused_features = F.normalize(reduced_features, p=2, dim=1)

        return cls_token, normalized_CNN_DINO_fused_features


if __name__ == "__main__":
    # 主程序，初始化模型并打印模型信息
    model = Dinov2Adapter(backbone_size="small").cuda()

    a = torch.randn(2, 3, 224, 224).cuda()
    cls_token, fused_features = model(a)
    print(cls_token.shape)
    print(fused_features.shape)

    # 计算模型参数总量，以及可训练参数数量，单位为M
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params / 1e6}M")
    print(f"Trainable parameters: {trainable_params / 1e6}M")
