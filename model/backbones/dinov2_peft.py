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


class Dinov2Peft(nn.Module):
    def __init__(self, backbone_size="small", output_dim=256):
        super().__init__()
        self.output_dim = output_dim
        self.backbone, self.checkpoint_path, self.patch_dim = self._initialize_backbone(
            backbone_size
        )
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
        # 定义线性层
        self.reduce_linear = nn.Linear(
            patch_dim[backbone_size[0]], self.output_dim, bias=False
        )

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
        cls_token=self.reduce_linear(cls_token)

        return cls_token


if __name__ == "__main__":
    model = Dinov2Peft().cuda()
    x = torch.randn(1, 3, 224, 224).cuda()
    y = model(x)
    print(y.shape)

