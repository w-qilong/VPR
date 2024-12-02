from transformers import Dinov2Model, Dinov2Config
import torch.nn as nn
import torch


class DinoV2Backbone(nn.Module):
    def __init__(self, freeze_layers=False):
        super().__init__()

        # 加载预训练的DINOv2模型
        self.dinov2 = Dinov2Model.from_pretrained(
            "/media/cartolab3/DataDisk/wuqilong_file/Projects/RerenkVPR/pretrained_model/dinov2_small"
        )

        if freeze_layers:
            # 冻结除最后两层外的所有参数
            self._freeze_base_layers()

        # 打印模型参数统计信息
        self.print_trainable_parameters()

    def _freeze_base_layers(self):
        """冻结基础模型参数，仅保留最后两层可训练"""
        # 首先冻结所有参数
        for param in self.dinov2.parameters():
            param.requires_grad = False

        # 解冻最后两层的参数
        for name, param in self.dinov2.named_parameters():
            # 通常最后两层在block.11和block.10中
            if "block.11" in name or "block.10" in name:
                param.requires_grad = True

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

    def forward(self, x):
        # 获取DINOv2的特征
        outputs = self.dinov2(x)
        # 使用CLS token的输出进行分类
        cls_token, patch_tokens = (
            outputs.last_hidden_state[:, 0],
            outputs.last_hidden_state[:, 1:],
        )

        return cls_token, patch_tokens


if __name__ == "__main__":
    model = DinoV2Backbone()
    print(model)
    # print(model)
    x = torch.randn(1, 3, 224, 224)
    y = model(x)
    print(y[0].shape)
    print(y[1].shape)
