import warnings

warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
from prettytable import PrettyTable
from dinov2.hub.backbones import (
    dinov2_vitb14,
    dinov2_vits14,
    dinov2_vitl14,
    dinov2_vitg14,
)

dinov2_backbones = {
    "dinov2_small": {
        "model": dinov2_vits14,
        "embedding_size": 384,
        "patch_size": 14,
        "num_heads": 6,
    },
    "dinov2_base": {
        "model": dinov2_vitb14,
        "embedding_size": 768,
        "patch_size": 14,
        "num_heads": 12,
    },
    "dinov2_large": {
        "model": dinov2_vitl14,
        "embedding_size": 1024,
        "patch_size": 14,
        "num_heads": 16,
    },
    "dinov2_giant": {
        "model": dinov2_vitg14,
        "embedding_size": 1536,
        "patch_size": 14,
        "num_heads": 24,
    },
}


class Dinov2Adapter(nn.Module):
    def __init__(self, backbone_size="dinov2_large", output_dim=256):
        """
        初始化Dinov2适配器
        Args:
            backbone_size: 主干网络大小 ('dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant')
            output_dim: 输出维度
        """
        super().__init__()
        
        if backbone_size not in dinov2_backbones:
            raise ValueError(f"不支持的backbone_size: {backbone_size}")
            
        self.output_dim = output_dim
        self.backbone_size = backbone_size
        
        # 初始化backbone并加载参数
        self._setup_model()
        
    def _setup_model(self):
        """设置模型并初始化所有必要的属性"""
        # 初始化backbone
        backbone_config = dinov2_backbones[self.backbone_size]
        self.model = backbone_config["model"](pretrained=False)
        
        # 设置模型属性
        self._set_model_attributes()
        
        # 加载预训练权重
        checkpoint_path = f"/home/cartolab3/.cache/torch/hub/checkpoints/dinov2_vit{self.backbone_size[7]}14_pretrain.pth"
        self._load_checkpoint(checkpoint_path)
        
        # 冻结参数并打印统计信息
        self._freeze_non_adapter_params()
        self._print_param_stats()
        
    def _set_model_attributes(self):
        """设置模型的关键属性"""
        self.num_features = self.model.num_features  # shape: [B, num_features]
        self.n_blocks = self.model.n_blocks
        self.num_heads = self.model.num_heads
        self.patch_size = self.model.patch_size
        self.num_register_tokens = self.model.num_register_tokens
        self.interpolate_antialias = self.model.interpolate_antialias
        self.interpolate_offset = self.model.interpolate_offset
        self.stride = self.model.patch_embed.proj.stride

    def _load_checkpoint(self, checkpoint_path):
        # 加载模型的检查点
        with open(checkpoint_path, "rb") as f:
            checkpoint = torch.load(f)
        backbone_dict = self.model.state_dict()
        backbone_dict.update(checkpoint.items())
        self.model.load_state_dict(backbone_dict)

    def _freeze_non_adapter_params(self):
        # 冻结backbone中不属于adapter的参数
        for name, param in self.model.named_parameters():
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
        """
        前向传播
        Args:
            x: 输入张量 shape: [B, 3, H, W]
        Returns:
            输出特征 shape: [B, num_features]
        """
        features = self.model.forward_features(x)  # shape: dict包含多个特征
        return features["x_norm_clstoken"]  # shape: [B, num_features]


if __name__ == "__main__":
    model = Dinov2Adapter(backbone_size="dinov2_large").cuda()
    print(model)
    x = torch.randn(1, 3, 224, 224).cuda()
    y = model(x)
    print(y.shape)

