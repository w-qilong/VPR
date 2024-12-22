import torch.nn as nn
from torch.hub import load
import torch.nn.functional as F
import torch
from typing import Union, List, Tuple

# DINOv2 模型配置字典，包含不同规模模型的参数
dinov2_model_configs = {
    "dinov2_small": {
        "name": "dinov2_vits14_reg",
        "embedding_size": 384,
        "patch_size": 14,
        "num_heads": 6,
    },
    "dinov2_base": {
        "name": "dinov2_vitb14_reg",
        "embedding_size": 768,
        "patch_size": 14,
        "num_heads": 12,
    },
    "dinov2_large": {
        "name": "dinov2_vitl14_reg",
        "embedding_size": 1024,
        "patch_size": 14,
        "num_heads": 16,
    },
    "dinov2_giant": {
        "name": "dinov2_vitg14_reg",
        "embedding_size": 1536,
        "patch_size": 14,
        "num_heads": 24,
    },
}

repo='/home/cartolab3/.cache/torch/hub/facebookresearch_dinov2_main'


def visualize_attention(image_tensor, attention_map, save_path=None, alpha=0.6):
    """
    可视化注意力图叠加到原始图像上
    Args:
        image_tensor: [3, H, W]
        attention_map: [num_heads, h, w]
        save_path: 保存路径
        alpha: 热图透明度，范围[0,1]
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.colors import LinearSegmentedColormap

    # 将注意力图上采样到原始图像大小
    attention_map = F.interpolate(
        attention_map.unsqueeze(0),
        size=image_tensor.shape[-2:],
        mode="bilinear",
        align_corners=False,
    )[0]

    # 处理原始图像
    img_u8 = image_tensor.cpu().numpy()
    img_u8 = (img_u8 - img_u8.min()) / (img_u8.max() - img_u8.min())
    img_u8 = (img_u8 * 255).astype(np.uint8).transpose(1, 2, 0)

    # 创建自定义热力图颜色映射（从透明到红色）
    colors = [(0, 0, 0, 0), (1, 0, 0, 1)]
    cmap = LinearSegmentedColormap.from_list("custom", colors, N=256)

    # 获取注意力头数量
    num_heads = attention_map.shape[0]

    # 创建子图
    fig, axes = plt.subplots(1, num_heads, figsize=(4 * num_heads, 4))
    if num_heads == 1:
        axes = [axes]

    # 为每个注意力头创建叠加图
    for h, ax in enumerate(axes):
        # 显示原始图像
        ax.imshow(img_u8)

        # 处理注意力图
        attn = attention_map[h].cpu()
        attn = (attn - attn.min()) / (attn.max() - attn.min())  # 归一化到[0,1]

        # 叠加热图
        ax.imshow(attn, cmap=cmap, alpha=alpha)
        ax.set_title(f"Attention Head {h}")
        ax.axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight", dpi=300)
        plt.close()
    else:
        plt.show()


class Dinov2Finetune(nn.Module):
    """DINOv2 backbone network for feature extraction and fine-tuning.

    This class wraps the DINOv2 model and provides functionality for:
    - Feature extraction from different layers and facets
    - Fine-tuning of selected layers
    - Dimension reduction of CLS token features
    - Saliency map generation

    Args:
        backbone_size (str): Model size from ['dinov2_small', 'dinov2_base', 'dinov2_large', 'dinov2_giant']
        finetune_last_n_layers (int): Number of last layers to fine-tune
        reduced_dim (int): Output dimension after CLS token reduction
    """

    def __init__(
        self,
        backbone_size: str = "dinov2_large",
        finetune_last_n_layers: int = 6,
        reduced_dim: int = 1024,
    ):
        super().__init__()

        if backbone_size not in dinov2_model_configs:
            raise ValueError(f"不支持的模型规模: {backbone_size}")

        # 加载预训练模型
        self.model = load(
            # repo_or_dir="facebookresearch/dinov2",
            repo_or_dir=repo,
            model=dinov2_model_configs[backbone_size]["name"],
            trust_repo=True,
            source='local'
        )

        # 模型属性
        self.num_features = self.model.num_features
        self.num_tokens = 1
        self.n_blocks = self.model.n_blocks
        self.num_heads = self.model.num_heads
        self.patch_size = self.model.patch_size
        self.num_register_tokens = self.model.num_register_tokens
        self.interpolate_antialias = self.model.interpolate_antialias
        self.interpolate_offset = self.model.interpolate_offset

        # 冻结除最后 n 层外的所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻最后 n 层的参数
        if finetune_last_n_layers > 0:
            for block in self.model.blocks[-finetune_last_n_layers:]:
                for param in block.parameters():
                    param.requires_grad = True

        # 打印模型参数信息
        self.print_trainable_parameters()

        # 定义线性层，用于对 cls token 进行降维
        self.cls_token_reducer = nn.Linear(self.num_features, reduced_dim)

        # 用于存储中间特征的列表
        self.intermediate_features = []
        self.hook_handlers = []

    def print_trainable_parameters(self):
        """打印模型参数统计信息"""
        from prettytable import PrettyTable

        table = PrettyTable()
        table.field_names = ["参数类型", "参数量"]

        # 计算总参数量
        total_params = sum(p.numel() for p in self.parameters())
        # 计算可训练参数量
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        # 计算冻结参数量
        frozen_params = total_params - trainable_params

        # 转换为M单位并添加数据行
        table.add_row(["总参数量", f"{total_params/1e6:.2f}M"])
        table.add_row(["可训练参数量", f"{trainable_params/1e6:.2f}M"])
        table.add_row(["冻结参数量", f"{frozen_params/1e6:.2f}M"])
        table.add_row(
            ["可训练参数比例", f"{100 * trainable_params / total_params:.2f}%"]
        )
        print(table)

    def _get_hook(self, feature_type: str):
        """Generate feature extraction hook function.

        Args:
            feature_type (str): One of ['attn', 'token', 'query', 'key', 'value']

        Returns:
            Callable: Hook function for feature extraction

        Raises:
            ValueError: If feature_type is not supported
        """
        if feature_type in ["attn", "token"]:

            def _hook(model, input, output):
                self.intermediate_features.append(output)

            return _hook

        feature_indices = {"query": 0, "key": 1, "value": 2}
        if feature_type not in feature_indices:
            raise ValueError(f"Unsupported feature type: {feature_type}")

        def _qkv_hook(module, input, output):
            # Extract QKV features from attention module
            input_tensor = input[0]  # Shape: (B, seq_len, hidden_dim)
            batch_size, seq_len, hidden_dim = input_tensor.shape

            # Reshape QKV tensor to separate heads
            qkv = (
                module.qkv(input_tensor)
                .reshape(
                    batch_size,
                    seq_len,
                    3,
                    module.num_heads,
                    hidden_dim // module.num_heads,
                )
                .permute(2, 0, 3, 1, 4)
            )  # Shape: (3, B, num_heads, seq_len, head_dim)

            # Extract specific feature (Q, K, or V)
            self.intermediate_features.append(qkv[feature_indices[feature_type]])

        return _qkv_hook

    def _register_hooks(self, layers: List[int], facet: str) -> None:
        """
        register hook to extract features.
        :param layers: layers from which to extract features.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        """
        for block_idx, block in enumerate(self.model.blocks):
            if block_idx in layers:
                if facet == "token":
                    self.hook_handlers.append(
                        block.register_forward_hook(self._get_hook(facet))
                    )
                elif facet == "attn":
                    self.hook_handlers.append(
                        block.attn.attn_drop.register_forward_hook(
                            self._get_hook(facet)
                        )
                    )
                elif facet in ["key", "query", "value"]:
                    self.hook_handlers.append(
                        block.attn.register_forward_hook(self._get_hook(facet))
                    )
                else:
                    raise TypeError(f"{facet} is not a supported facet.")

    def _unregister_hooks(self) -> None:
        """
        unregisters the hooks. should be called after feature extraction.
        """
        for handle in self.hook_handlers:
            handle.remove()
        self.hook_handlers = []

    def _extract_features(
        self, batch: torch.Tensor, layers: List[int] = [11], facet: str = "key"
    ) -> List[torch.Tensor]:
        """
        extract features from the model
        :param batch: batch to extract features for. Has shape BxCxHxW.
        :param layers: layer to extract. A number between 0 to 11.
        :param facet: facet to extract. One of the following options: ['key' | 'query' | 'value' | 'token' | 'attn']
        :return : tensor of features.
                if facet is 'key' | 'query' | 'value' has shape Bxhxtxd
                if facet is 'attn' has shape Bxhxtxt
                if facet is 'token' has shape Bxtxd
        """
        B, C, H, W = batch.shape
        self._feats = []
        self._register_hooks(layers, facet)
        _ = self.model(batch)
        self._unregister_hooks()
        return self._feats

    def extract_saliency_maps(self, batch: torch.Tensor, layers=[11]) -> torch.Tensor:
        """
        extract saliency maps. The saliency maps are extracted by averaging several attention heads from the last layer
        in of the CLS token. All values are then normalized to range between 0 and 1.
        :param batch: batch to extract saliency maps for. Has shape BxCxHxW.
        :return: a tensor of saliency maps. has shape Bxt-1
        """
        self._extract_features(batch, layers, "attn")
        curr_feats = self._feats[0]  # Bxhxtxt
        cls_attn_map = curr_feats[:, :, 0, self.num_register_tokens + 1 :].mean(
            dim=1
        )  # Bx(t-1)
        temp_mins, temp_maxs = cls_attn_map.min(dim=1)[0], cls_attn_map.max(dim=1)[0]
        cls_attn_maps = (cls_attn_map - temp_mins) / (
            temp_maxs - temp_mins
        )  # normalize to range [0,1]
        return cls_attn_maps

    def forward(
        self,
        x: torch.Tensor,
        is_training: bool = True,
        layers: List[int] = [11],
        facet: str = "value",
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through the model.

        Args:
            x (torch.Tensor): Input images of shape (B, 3, H, W)
            is_training (bool): Whether in training mode
            layers (List[int]): Layer indices to extract features from
            facet (str): Feature type to extract, one of ['key', 'query', 'value', 'token', 'attn']

        Returns:
            If is_training:
                torch.Tensor: Reduced CLS token features of shape (B, reduced_dim)
            Else:
                Tuple[torch.Tensor, torch.Tensor]: (
                    Reduced CLS token features of shape (B, reduced_dim),
                    Intermediate features shape depends on facet type
                )
        """
        if not is_training:
            self.intermediate_features = []
            self._register_hooks(layers=layers, facet=facet)

        # Extract features using DINOv2 backbone
        x = self.model.forward_features(x)

        # Get normalized tokens
        x = x["x_norm_clstoken"]

        # Reduce CLS token dimension
        # x = self.cls_token_reducer(x)

        if not is_training:
            self._unregister_hooks()
            return x, self.intermediate_features[0]
        return x


if __name__ == "__main__":
    from PIL import Image
    from torchvision import transforms as T

    # 加载图像
    img_path = "/media/cartolab3/DataDisk/wuqilong_file/Projects/RerenkVPR/imgs/output_imgs/tokyo/00010.jpg"
    img = Image.open(img_path).convert("RGB")

    image_size_eval = (224, 224)
    mean_std = {"mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}
    valid_transform = T.Compose(
        [
            T.Resize(image_size_eval, interpolation=T.InterpolationMode.BILINEAR),
            T.ToTensor(),
            T.Normalize(mean=mean_std["mean"], std=mean_std["std"]),
        ]
    )

    x = valid_transform(img=img)
    x = x.unsqueeze(0)
    print(x.shape)

    # 初始化模型
    model = Dinov2Finetune(backbone_size='dinov2_large',finetune_last_n_layers=6, reduced_dim=1024).cuda()
    print(model)
    print(model.num_register_tokens)

    # 前向传播
    output ,attn = model(x.cuda(), is_training=False, layers=[23], facet="value")
    print(output.shape)
    print(attn.shape)

    # 可视化attention head，排除register tokens，【cls, register, patch】
    # h= int((image_size_eval/14)**0.5)
    # attn = attn[:, :, 0, 5:][0].reshape(model.num_heads, h, h)
    # visualize_attention(
    #     image_tensor=x[0].detach().cpu(),
    #     attention_map=attn.detach().cpu(),
    #     save_path=None,
    #     alpha=0.8,
    # )
