import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load

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


class LowRankSideAdapter(nn.Module):
    def __init__(self, input_dim, hidden_dim, alpha=1.0):
        """
        Low-rank side adapter for feature refinement.
        Args:
            input_dim: Input feature dimensionality.
            hidden_dim: Hidden layer dimensionality (reduced rank).
            alpha: Scaling factor for the adapter's output.
        """
        super(LowRankSideAdapter, self).__init__()
        self.down_proj = nn.Linear(input_dim, hidden_dim)
        self.up_proj = nn.Linear(hidden_dim, input_dim)
        self.alpha = alpha

    def forward(self, x):
        """
        Forward pass through the adapter.
        Args:
            x: Input tensor of shape [batch_size, seq_len, input_dim].
        Returns:
            Adapted tensor of the same shape as input.
        """
        x = self.down_proj(x) # batch_size, seq_len, hidden_dim
        x = F.gelu(x)
        x = self.up_proj(x)
        return self.alpha * x


class Dinov2Losa(nn.Module):
    def __init__(
        self,
        backbone_size="dinov2_large",
        input_dim=1024,
        hidden_dim=16,
        alpha=1.0,
        output_dim=1024,
    ):
        """
        LoSA fine-tuning for DINO V2 with intermediate layer features.
        Args:
            backbone_size: Pre-trained DINO V2 model (frozen).
            num_layers: Number of LoSA layers to process intermediate features.
            input_dim: Dimensionality of DINO features.
            hidden_dim: Dimensionality of the hidden layer in LoSA.
            alpha: Scaling factor for LoSA.
            output_dim: Number of classes for the downstream task.
        """
        super(Dinov2Losa, self).__init__()

        # 加载预训练模型# backbone 参数
        self.backbone_size = backbone_size
        # 加载预训练模型
        self._load_model()

        # adapter属性
        self.num_layers = len(self.model.blocks)
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.alpha = alpha
        self.output_dim = output_dim

        # DINO V2模型属性
        self.num_features = self.model.num_features
        self.n_blocks = self.model.n_blocks
        self.num_heads = self.model.num_heads
        self.patch_size = self.model.patch_size
        self.num_register_tokens = self.model.num_register_tokens
        self.interpolate_antialias = self.model.interpolate_antialias
        self.interpolate_offset = self.model.interpolate_offset
        self.stride = self.model.patch_embed.proj.stride

        self.adapters= nn.ModuleList(
            [
                LowRankSideAdapter(input_dim, hidden_dim, self.alpha)
                for _ in range(self.num_layers)
            ]
        )

        self.task_head = nn.Linear(input_dim, self.output_dim)  # output_dim

    def _load_model(self):
        """加载预训练模型"""
        self.model = load(
            # repo_or_dir="facebookresearch/dinov2",
            repo_or_dir="/home/cartolab3/.cache/torch/hub/facebookresearch_dinov2_main",
            model=dinov2_backbones[self.backbone_size]["name"],
            trust_repo=True,
            source="local",
        )

    def forward(self, x):
        """
        Forward pass through the model.
        Args:
            x: Input images of shape [batch_size, channels, height, width].
        Returns:
            Logits for the downstream task.
        """

        # 准备输入tokens
        x = self.model.prepare_tokens_with_masks(x)

        backbone_outputs = []

        # Extract intermediate features from frozen backbone
        with torch.no_grad():
            for idx, blk in enumerate(self.model.blocks):
                x = blk(x)
                x = x.detach()
                backbone_outputs.append(x) # batch_size, seq_len, input_dim

        # Apply LoSA adapters on intermediate layers
        y = backbone_outputs[-1]  # Start with the last layer's output
        for i, adapter in enumerate(self.adapters):
            # Select the corresponding intermediate layer features
            backbone_feature = backbone_outputs[i] # batch_size, seq_len, input_dim
            y = adapter(backbone_feature + y) + y
 
        # Task-specific head
        logits = self.task_head(y[:, 0])  # Use mean pooling along seq_len for classification
        return logits


# Example Usage
if __name__ == "__main__":
    # Load DINO V2 backbone from Hugging Face
    backbone = Dinov2Losa(
        backbone_size="dinov2_large",
        input_dim=1024,
        hidden_dim=64,
        alpha=1.0,
        output_dim=1024,
    ).cuda()  # Replace with actual model name
    backbone.eval()  # Freeze the backbone

    # Input data
    batch_size = 1
    channels, height, width = 3, 224, 224
    x = torch.rand(batch_size, channels, height, width).cuda()

    # Forward pass
    logits = backbone(x)
    print("Logits shape:", logits.shape)  # Expected: [batch_size, task_classes]

