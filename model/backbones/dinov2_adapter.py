from functools import partial
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
    def __init__(self, backbone_size="small"):
        super().__init__()

        self.patch_dim = {"small": 384, "base": 768, "large": 1024, "giant": 1024}
        self.embed_dim = self.patch_dim[backbone_size]

        self.backbone, self.checkpoint_path = self._initialize_backbone(backbone_size)
        print(self.backbone)
        # 加载检查点
        self._load_checkpoint(self.checkpoint_path)
        # 冻结非adapter的参数
        self._freeze_non_adapter_params()
        # 打印模型参数统计信息
        self._print_param_stats()

    def _initialize_backbone(self, backbone_size):
        # 根据backbone_size选择合适的backbone，并返回其对应的检查点路径
        backbone_map = {
            "small": dinov2_vits14,
            "base": dinov2_vitb14,
            "large": dinov2_vitl14,
            "giant": dinov2_vitg14,
        }

        checkpoint_path = f"/home/cartolab3/.cache/torch/hub/checkpoints/dinov2_vit{backbone_size[0]}14_pretrain.pth"
        return (
            backbone_map[backbone_size](pretrained=False),
            checkpoint_path,
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
        return cls_token, patch_tokens


from modules import SpatialPriorModule, InteractionBlock, deform_inputs, MSDeformAttn
from timm.models.layers import trunc_normal_
from torch.nn.init import normal_
import math


class Dinov2Adapter(Dinov2Peft):
    def __init__(
        self,
        backbone_size="small",
        pretrain_size=224,
        conv_inplane=64,
        deform_num_heads=6,
        n_points=4,
        init_values=0.0,
        interaction_indexes=[[0, 2], [3, 5], [6, 8], [9, 11]],
        with_cffn=True,
        cffn_ratio=0.25,
        deform_ratio=1.0,
        drop_path_rate=0.0,
        add_vit_feature=True,
        use_extra_extractor=True,
        with_cp=False,
    ):
        super().__init__(backbone_size)

        # 初始化模型参数
        self.interaction_indexes = interaction_indexes
        self.add_vit_feature = add_vit_feature
        self.norm_layer = partial(nn.LayerNorm, eps=1e-6)
        embed_dim = self.embed_dim

        # 初始化blocks
        self.blocks = self.backbone.blocks

        # 初始化模块
        self.level_embed = nn.Parameter(torch.zeros(3, self.patch_dim[backbone_size]))
        embed_dim = self.embed_dim
        self.spm = SpatialPriorModule(
            inplanes=conv_inplane, embed_dim=embed_dim, with_cp=False
        )
        self.interactions = nn.Sequential(
            *[
                InteractionBlock(
                    dim=embed_dim,
                    num_heads=deform_num_heads,
                    n_points=n_points,
                    init_values=init_values,
                    drop_path=drop_path_rate,
                    norm_layer=self.norm_layer,
                    with_cffn=with_cffn,
                    cffn_ratio=cffn_ratio,
                    deform_ratio=deform_ratio,
                    extra_extractor=(
                        (True if i == len(interaction_indexes) - 1 else False)
                        and use_extra_extractor
                    ),
                    with_cp=with_cp,
                )
                for i in range(len(interaction_indexes))
            ]
        )
        self.up = nn.ConvTranspose2d(embed_dim, embed_dim, 2, 2)
        self.norm1 = nn.SyncBatchNorm(embed_dim)
        self.norm2 = nn.SyncBatchNorm(embed_dim)
        self.norm3 = nn.SyncBatchNorm(embed_dim)
        self.norm4 = nn.SyncBatchNorm(embed_dim)

        self.up.apply(self._init_weights)
        self.spm.apply(self._init_weights)
        self.interactions.apply(self._init_weights)
        self.apply(self._init_deform_weights)
        normal_(self.level_embed)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm) or isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _init_deform_weights(self, m):
        if isinstance(m, MSDeformAttn):
            m._reset_parameters()

    def _add_level_embed(self, c2, c3, c4):
        c2 = c2 + self.level_embed[0]
        c3 = c3 + self.level_embed[1]
        c4 = c4 + self.level_embed[2]
        return c2, c3, c4

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # SPM forward
        c1, c2, c3, c4 = self.spm(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding forward
        x = self.backbone.prepare_tokens_with_masks(x)
        bs, n, dim = x.shape
        x = x[:, 1:]
        H, W = math.ceil(x.size(-2) ** 0.5), math.ceil(x.size(-2) ** 0.5) 

        # Interaction
        outs = list()
        for i, layer in enumerate(self.interactions):
            print(f"Shape of x after prepare_tokens_with_masks: {x.shape}")
            print(f"Shape of c after SPM: {c.shape}")
            indexes = self.interaction_indexes[i]
            x, c = layer(
                x,
                c,
                self.blocks[indexes[0] : indexes[-1] + 1],
                deform_inputs1,
                deform_inputs2,
                H,
                W,
            )
            outs.append(x.transpose(1, 2).view(bs, dim, H, W).contiguous())


if __name__ == "__main__":
    model = Dinov2Adapter(backbone_size="small")

    x = torch.randn(2, 3, 224, 224)
    y = model(x)
