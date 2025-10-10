import torch
import torch.nn as nn


class Dinov3Backbone(nn.Module):

    def __init__(self, finetune_last_n_layers=2):
        super(Dinov3Backbone, self).__init__()
        # Define the layers for the backbone here
        REPO_DIR = "model/dinov3"
        weight_path = "/home/cartolab3/.cache/torch/hub/checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
        self.model = torch.hub.load(
            REPO_DIR, "dinov3_vitl16", source="local", weights=weight_path
        )
        self.finetune_last_n_layers = finetune_last_n_layers
        self.n_storage_tokens = self.model.n_storage_tokens
        print(self.n_storage_tokens)

    def forward(self, x, mask=None):
        """前向：
        - 冻结的前面layers用 no_grad 前传；
        - 最后 N 个 blocks 可训练；
        - 输出CLS向量 (B, C)。
        """
        # 准备输入tokens与空间尺寸
        x, (H, W) = self.model.prepare_tokens_with_masks(x, mask)

        # 前向传播
        with torch.no_grad():
            for idx, blk in enumerate(
                self.model.blocks[: -self.finetune_last_n_layers]
            ):
                rope_sincos = self.model.rope_embed(H=H, W=W)
                x = blk(x, rope_sincos)
        x = x.detach()

        for blk in self.model.blocks[-self.finetune_last_n_layers :]:
            rope_sincos = self.model.rope_embed(H=H, W=W)
            x = blk(x, rope_sincos)

        x = self.model.norm(x)
        x = x[:, : self.model.n_storage_tokens + 1]
        cls_token = x[:, 0]
        return cls_token


if __name__ == "__main__":
    model = Dinov3Backbone()

    a = model(torch.randn(3, 3, 480, 480))
    print(a.shape)
