import torch.nn as nn
from torch.hub import load
import torch.nn.functional as F
import torch

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
        self, backbone_size="dinov2_large", finetune_last_n_layers=1, reduced_dim=1024
    ):
        super().__init__()

        # 初始化配置
        self._init_config(backbone_size, finetune_last_n_layers, reduced_dim)
        # 加载预训练模型
        self._load_model()
        # 设置参数冻结
        # self._freeze_layers()
        # 打印参数统计
        # self.print_trainable_parameters()

        # 初始化attention相关属性
        self.attention_outputs = {}
        self.hook_handle = None

    def _init_config(self, backbone_size, finetune_last_n_layers, reduced_dim):
        """初始化模型配置"""
        if backbone_size not in dinov2_backbones:
            raise ValueError(f"Invalid backbone_size: {backbone_size}")

        self.backbone_size = backbone_size
        self.finetune_last_n_layers = finetune_last_n_layers
        self.reduced_dim = reduced_dim

        config = dinov2_backbones[backbone_size]
        self.num_heads = config["num_heads"]
        self.patch_size = config["patch_size"]
        self.embedding_size = config["embedding_size"]

        # 初始化降维层
        self.reduce_linear = nn.Linear(self.embedding_size, reduced_dim, bias=False)

    def _load_model(self):
        """加载预训练模型"""
        self.model = load(
            repo_or_dir="/home/cartolab3/.cache/torch/hub/facebookresearch_dinov2_main",
            model=dinov2_backbones[self.backbone_size]["name"],
            source='local',
            trust_repo=True,
        )

    def _freeze_layers(self):
        """冻结模型参数，仅保留指定数量的可训练层"""
        # 首先冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False

        # 解冻最后n层
        if self.finetune_last_n_layers > 0:
            start_block = len(self.model.blocks) - self.finetune_last_n_layers
            for name, param in self.model.named_parameters():
                if name.startswith(f"blocks.{start_block}"):
                    param.requires_grad = True

    def forward(self, x, masks=None, is_training=False, block_idx=-1):
        """模型前向传播

        Args:
            x (torch.Tensor): 输入图像 [B, C, H, W]
            masks (torch.Tensor, optional): 注意力掩码
            is_training (bool): 是否为训练模式
            block_idx (int): 获取attention的层索引(-1表示最后一层)

        Returns:
            训练模式: cls_token [B, reduced_dim]
            测试模式: (cls_token, (q, k, v))
        """
        # 准备输入tokens
        x = self.model.prepare_tokens_with_masks(x, masks)
        B, N, C = x.shape

        # 测试模式下设置attention hook
        if not is_training:
            target_idx = block_idx if block_idx >= 0 else len(self.model.blocks) - 1
            self._register_hook(target_idx)
            self.attention_outputs.clear()

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
        # cls_token = self.reduce_linear(x[:, 0, :])
        cls_token = x[:, 0]

        if is_training:
            return cls_token

        # 测试模式处理attention输出
        return self._process_test_output(cls_token, B, N, C)

    def _process_test_output(self, cls_token, B, N, C):
        """处理测试模式的输出"""
        qkv = self.attention_outputs["qkv"]
        qkv = qkv.reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(
            2, 0, 3, 1, 4
        )
        q, k, v = qkv[0], qkv[1], qkv[2]

        # 清理hook
        if self.hook_handle is not None:
            self.hook_handle.remove()
            self.hook_handle = None

        return cls_token, (q, k, v)

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

    def _attention_hook(self):
        """定义hook函数来捕获attention的q,k,v"""

        def hook(module, input, output):
            self.attention_outputs = {
                "qkv": output,
            }

        return hook

    def _register_hook(self, block_idx):
        """为指定block的attention层注册hook
        Args:
            block_idx (int): block的索引
        """
        if not 0 <= block_idx < len(self.model.blocks):
            raise ValueError(
                f"Invalid block index {block_idx}. Should be in range [0, {len(self.model.blocks)-1}]"
            )

        # 如果已存在hook，先移除
        if self.hook_handle is not None:
            self.hook_handle.remove()

        # 注册新的hook
        target_block = self.model.blocks[block_idx]
        self.hook_handle = target_block.attn.qkv.register_forward_hook(
            self._attention_hook()
        )


if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image
    import torch

    model = Dinov2Backbone(
        backbone_size="dinov2_base", finetune_last_n_layers=1, reduced_dim=1024
    ).cuda()

    print(model)

    x = torch.rand(2, 3, 224, 224).cuda()

    # 训练模式
    output = model(x, is_training=True)  # 只返回cls_token

    # 测试模式，获取第5个block的attention输出
    cls_token, qkv = model(x, is_training=False, block_idx=10)

    # 测试模式，获取最后一个block的attention输出
    cls_token, (q, k, v) = model(x, is_training=False)  # block_idx默认为-1

    print("Cls token shape:", cls_token.shape)
    print("Patch tokens shape:", q.shape)
