import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AggregateBlock(nn.Module):
    """A MLP-based mixing block that processes tokens while maintaining their dimensionality.

    Args:
        token_dim (int): Number of input tokens/dimension to mix
        mlp_ratio (float): Expansion ratio for the hidden layer dimension

    Shape:
        - Input: (B, C, N) where B is batch size, C is channels, N is number of tokens
        - Output: (B, C, N)
    """

    def __init__(self, token_dim, mlp_ratio=1):
        super().__init__()
        hidden_dim = int(token_dim * mlp_ratio)
        self.mix = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, token_dim),
        )
        self._init_weights()

    def _init_weights(self):
        """Initialize the weights using truncated normal distribution."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        return x + self.mix(x)  # Residual connection


class Aggregator(nn.Module):
    """Token mixing and dimension reduction module.

    This module performs token mixing through multiple AggregateBlocks and then
    projects the features to desired output dimensions, finally normalizing the output.

    Args:
        feat_dim (int): Input feature dimension (default: 768)
        num_tokens (int): Number of input tokens (default: 256)
        out_dim (int): Output feature dimension (default: 768)
        mix_depth (int): Number of mixing blocks (default: 4)
        mlp_ratio (float): Expansion ratio in mixing blocks (default: 4)
        out_tokens (int): Number of output tokens after reduction (default: 3)

    Shape:
        - Input: (B, N, C) where B is batch size, N is num_tokens, C is feat_dim
        - Output: (B, out_dim * out_tokens)
    """

    def __init__(
        self,
        feat_dim=768,
        num_tokens=256,
        out_dim=768,
        mix_depth=4,
        mlp_ratio=4,
        out_tokens=3,
    ):
        super().__init__()

        self.mix = nn.Sequential(
            *[
                AggregateBlock(token_dim=num_tokens, mlp_ratio=mlp_ratio)
                for _ in range(mix_depth)
            ]
        )
        self.feat_proj = nn.Linear(feat_dim, out_dim)
        self.token_proj = nn.Linear(num_tokens, out_tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Token mixing
        x = x.permute(0, 2, 1)  # [B,F,N]
        x = self.mix(x)  # [B,F,N]

        # Dimension reduction
        x = x.permute(0, 2, 1)  # [B,N,F]
        x = self.feat_proj(x)  # [B,N,out_dim]
        x = x.permute(0, 2, 1)  # [B,out_dim,N]
        x = self.token_proj(x)  # [B,out_dim,out_tokens]

        # L2 normalization
        return F.normalize(x.flatten(1), p=2, dim=-1)


def print_nb_params(m):
    model_parameters = filter(lambda p: p.requires_grad, m.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    print(f"Trainable parameters: {params/1e6:.3}M")


if __name__ == "__main__":

    a = torch.randn((1, 529, 1024))
    agg = Aggregator(
        feat_dim=1024,
        num_tokens=529,
        out_dim=1024,
        mix_depth=6,
        mlp_ratio=2,
        out_tokens=4,
    )
    print_nb_params(agg)
    out = agg(a)
    print(out.shape)
