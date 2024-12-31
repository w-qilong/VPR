import torch.nn as nn
from torch.hub import load
import torch.nn.functional as F
import torch
from xformers.ops import unbind, memory_efficient_attention

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


class Dinov2Backbone(nn.Module):
    """DINOv2 backbone with configurable fine-tuning and dimension reduction

    Attributes:
        backbone_size (str): Model size (small/base/large/giant)
        finetune_last_n_layers (int): Number of layers to fine-tune
        reduced_dim (int): Output dimension after reduction
        num_heads (int): Number of attention heads
        patch_size (int): Size of image patches
        embedding_size (int): Size of embeddings before reduction
    """

    def __init__(
        self, backbone_size="dinov2_large", finetune_last_n_layers=1):
        super().__init__()

        # backbone 参数
        self.backbone_size= backbone_size
        self.finetune_last_n_layers=finetune_last_n_layers

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

    def forward(self, x):
        # 准备输入tokens
        x = self.model.prepare_tokens_with_masks(x)

        # 前向传播
        with torch.no_grad():
            for idx, blk in enumerate(
                self.model.blocks[: -self.finetune_last_n_layers]
            ):
                x = blk(x)
        x = x.detach()

        for blk in self.model.blocks[-self.finetune_last_n_layers :]:
            x = blk(x)

        # 获取最终输出
        x = self.model.norm(x)
        x = x[:, 0]

        return x


if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image
    import torch

    model = Dinov2Backbone(backbone_size="dinov2_large", finetune_last_n_layers=1).cuda()

    print(model)

    x = torch.rand(2, 3, 224, 224).cuda()

    # 测试模式，获取第5个block的attention输出
    cls_token= model(x)

    print("Cls token shape:", cls_token.shape)

