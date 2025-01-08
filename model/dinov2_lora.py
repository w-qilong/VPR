import torch.nn as nn
from torch.hub import load
import torch.nn.functional as F
import torch
from peft import LoraConfig, get_peft_model

dinov2_backbones = {
    "dinov2_small": {
        "name": "dinov2_vits14",
        "embedding_size": 384,
        "patch_size": 14,
        "num_heads": 6,
    },
    "dinov2_base": {
        "name": "dinov2_vitb14",
        "embedding_size": 768,
        "patch_size": 14,
        "num_heads": 12,
    },
    "dinov2_large": {
        "name": "dinov2_vitl14",
        "embedding_size": 1024,
        "patch_size": 14,
        "num_heads": 16,
    },
    "dinov2_giant": {
        "name": "dinov2_vitg14",
        "embedding_size": 1536,
        "patch_size": 14,
        "num_heads": 24,
    },
}

class Dinov2Lora(nn.Module):
    """DINOv2 backbone with configurable fine-tuning and dimension reduction

    Attributes:
        backbone_size (str): Model size (small/base/large/giant)
        finetune_last_n_layers (int): Number of layers to fine-tune
        reduced_dim (int): Output dimension after reduction
        num_heads (int): Number of attention heads
    """

    def __init__(
        self, backbone_size="dinov2_large",  # 主干网络大小，可选：dinov2_small/base/large/giant
        lora_r=4,                            # LoRA的秩，控制可训练参数的数量，较小的值意味着更少的参数
        lora_alpha=32,                       # LoRA的缩放因子，用于控制LoRA更新的影响程度。控制LORA参数与原始参数的比率
        lora_dropout=0.1                     # LoRA层的dropout率，用于防止过拟合
    ):
        super().__init__()

        # backbone 参数
        self.backbone_size= backbone_size
        self.lora_r=lora_r
        self.lora_alpha=lora_alpha
        self.lora_dropout=lora_dropout

        # 加载预训练模型
        self._load_model()

        # DINO V2模型属性
        self.num_features = self.model.num_features
        self.n_blocks = self.model.n_blocks
        self.num_heads = self.model.num_heads
        self.patch_size = self.model.patch_size
        self.num_register_tokens = self.model.num_register_tokens
        self.interpolate_antialias = self.model.interpolate_antialias
        self.interpolate_offset = self.model.interpolate_offset
        self.stride = self.model.patch_embed.proj.stride

    def _load_model(self):
        """加载预训练模型"""
        self.model = load(
            # repo_or_dir="facebookresearch/dinov2",
            repo_or_dir='/home/cartolab3/.cache/torch/hub/facebookresearch_dinov2_main',
            model=dinov2_backbones[self.backbone_size]["name"],
            trust_repo=True,
            source='local',
        )

        # 配置LoRA
        self.lora_config = LoraConfig(
            r=self.lora_r,
            lora_alpha=self.lora_alpha,
            target_modules=["qkv"],  # 更具体的模块路径
            lora_dropout=self.lora_dropout,
            bias="none",
        )

        # 将模型转换为LoRA模型
        self.model = get_peft_model(self.model, self.lora_config)

        self.model.print_trainable_parameters()

    def forward(self, x):
        x = self.model.prepare_tokens_with_masks(x)

        for blk in self.model.blocks:
            x = blk(x)

        x_norm = self.model.norm(x)

        return x_norm[:, 0]


if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image
    import torch

    model = Dinov2Lora(backbone_size="dinov2_large").cuda()
    print(model)

    x = torch.rand(2, 3, 224, 224).cuda()

    # 测试模式，获取第5个block的attention输出
    cls_token= model(x)

    print("Cls token shape:", cls_token.shape)

