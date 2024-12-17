import torch
from torch import nn
from .backbones import DinoV2Backbone, ResNetFPN_8_2, ResNetFPN_16_4
import torch.nn.functional as F

# 为细粒度CNN特征提取配置参数
configs = {
    "ResNetFPN_8_2": {"initial_dim": 128, "block_dims": [128, 196, 256]},
    "ResNetFPN_16_4": {"initial_dim": 128, "block_dims": [128, 196, 256, 384]},
}

# 定义不同DINOv2 Backbone的输出维度
dino_v2_backbone_output_dims = {
    "small": 384,
    "base": 768,
    "large": 1024,
}


# 特征提取器
class FeatureExtractor(nn.Module):
    def __init__(
        self, backbone_size="small", fine_feature_extractor="ResNetFPN_8_2", finetune_last_n_layers=1, reduce_dim=128, top_k=64):
        super(FeatureExtractor, self).__init__()

        # 配置参数
        self.backbone_size = backbone_size
        self.finetune_last_n_layers = finetune_last_n_layers
        self.fine_feature_extractor = fine_feature_extractor
        self.reduce_dim = reduce_dim
        self.top_k = top_k

        # 粗粒度特征提取器
        self.corse_feature_extractor = DinoV2Backbone(
            backbone_size=backbone_size,
            finetune_last_n_layers=finetune_last_n_layers,
        )

        # 细粒度特征提取器
        if fine_feature_extractor == "ResNetFPN_8_2":
            self.fine_feature_extractor = ResNetFPN_8_2(
                configs["ResNetFPN_8_2"],
            )
        elif fine_feature_extractor == "ResNetFPN_16_4":
            self.fine_feature_extractor = ResNetFPN_16_4(
                configs["ResNetFPN_16_4"],
            )

        # 分别定义用于降低fine_feature和fine_patch_tokens维度的线性层
        coarse_patch_dim = dino_v2_backbone_output_dims[backbone_size]
        if fine_feature_extractor == "ResNetFPN_8_2":
            fine_patch_dim = 128
        elif fine_feature_extractor == "ResNetFPN_16_4":
            fine_patch_dim = 128
                
        # 使用Sequential组合投影层和归一化
        self.projection = nn.Sequential(
            nn.Conv2d(coarse_patch_dim + fine_patch_dim, reduce_dim, kernel_size=1),
            nn.BatchNorm2d(reduce_dim),
            nn.ReLU(inplace=True)
        )

    def _process_features(self, local_features, patch_tokens, attn):
        """处理和融合特征的辅助函数"""
        B, C, H, W = local_features.shape
        
        # 重塑并插值patch tokens
        h = int(patch_tokens.shape[1]**0.5)
        patch_tokens = patch_tokens.transpose(1, 2).reshape(B, -1, h, h)
        patch_tokens = F.interpolate(patch_tokens, 
                                   size=(H, W), 
                                   mode='bilinear', 
                                   align_corners=False)

        # 融合特征并投影
        combined_features = torch.cat([patch_tokens, local_features], dim=1)
        projected_features = self.projection(combined_features)
        
        # 处理注意力图
        attn = F.interpolate(attn, size=(H, W), mode='bilinear', align_corners=False)
        attn = attn.mean(dim=1, keepdim=True)  # 平均池化所有注意力头

        # 获取top-k索引
        attn_flat = attn.view(B, -1)
        topk_indices = torch.topk(attn_flat, self.top_k, dim=-1).indices

        # 提取对应特征
        features_flat = projected_features.view(B, C, -1)
        topk_features = features_flat.gather(2, topk_indices.unsqueeze(1).expand(-1, C, -1))
        
        # L2归一化
        topk_features = F.normalize(topk_features, p=2, dim=1)

        return topk_features

    def forward(self, x):
        # DINOv2提取粗粒度特征
        attn, cls_token, patch_tokens = self.corse_feature_extractor(x)

        # CNN提取细粒度特征
        fine_feature = self.fine_feature_extractor(x)[1]  # [B, C, H, W]

        # 获取top-k细粒度特征，用于局部特征匹配
        topk_features = self._process_features(fine_feature, patch_tokens, attn)

        return cls_token, topk_features  # [B, N, C], [B, reduce_dim, H, W]



if __name__ == "__main__":
    # 检查是否有可用的GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 特征提取器测试
    feature_extractor = FeatureExtractor(backbone_size="small", finetune_last_n_layers=1).to(device)
    batch_size = 16
    x1 = torch.randn(batch_size, 3, 224, 224).to(device)
    x2 = torch.randn(batch_size, 3, 224, 224).to(device)
    
    # 提取两张图片的特征
    cls_token1, fine_patch_tokens1 = feature_extractor(x1)
    cls_token2, fine_patch_tokens2 = feature_extractor(x2)
    
    print("特征提取器输出:")
    print("全局特征形状:", cls_token1.shape)
    print("局部特征形状:", fine_patch_tokens1.shape)
