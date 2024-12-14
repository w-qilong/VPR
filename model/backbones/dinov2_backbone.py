from transformers import Dinov2Model, Dinov2Config
import torch.nn as nn
import torch
import torch.nn.functional as F


class DinoV2Backbone(nn.Module):
    def __init__(self, backbone_size, finetune_last_n_layers=2):
        super().__init__()

        # 加载预训练的DINOv2模型
        self.model = Dinov2Model.from_pretrained(
            f"D:\PythonProject\RerankVPR\pretrained_model\dinov2_{backbone_size}"
        )

        # 冻结所有参数
        self.freeze_layers(finetune_last_n_layers)

    def freeze_layers(self,finetune_last_n_layers):
        """冻结除了最后n层之外的所有层"""
        # 首先冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
            
        # 获取总层数
        total_layers = len(self.model.encoder.layer)
        
        # 解冻最后n层
        for i in range(total_layers - finetune_last_n_layers, total_layers):
            for param in self.model.encoder.layer[i].parameters():
                param.requires_grad = True
                
        # 打印参数状态
        self.print_trainable_parameters()

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

    def get_patch_size(self):
        """获取patch size"""
        return self.model.config.patch_size

    def forward(self, x):
        # 获取输入图像的尺寸
        B, C, H, W = x.shape
        patch_size = self.get_patch_size()
        h = H // patch_size
        w = W // patch_size

        # 前向传播，获取最后一层的注意力和特征
        outputs = self.model(x, output_attentions=True, return_dict=True)

        # 获取最后一层的注意力 [B, num_heads, N, N]
        last_attn = outputs.attentions[-1]

        # 获取patch tokens [B, N, C]
        patch_tokens = outputs.last_hidden_state

        # 处理注意力图
        B, H, N, _ = last_attn.shape

        # 移除CLS token的注意力
        attn = last_attn[:, :, 1:, 1:]

        # 计算平均注意力并重塑为空间形式
        attn = attn.mean(dim=-1)  # [B, num_heads, N-1]
        attn = attn.reshape(B, H, h, w)

        return attn, patch_tokens[:, 0], patch_tokens[:, 1:]  # 移除CLS token
    
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
        mode='bilinear',
        align_corners=False
    )[0]
    
    # 处理原始图像
    if image_tensor.max() <= 1:
        image_tensor = image_tensor * 255
    image_np = image_tensor.permute(1, 2, 0).cpu().numpy().astype(np.uint8)
    
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
        ax.imshow(image_np)
        
        # 处理注意力图
        attn = attention_map[h].cpu()
        attn = (attn - attn.min()) / (attn.max() - attn.min())  # 归一化到[0,1]
        
        # 叠加热图
        ax.imshow(attn, cmap=cmap, alpha=alpha)
        ax.set_title(f'Attention Head {h}')
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close()
    else:
        plt.show()


        
 
if __name__ == "__main__":
    from torchvision import transforms
    from PIL import Image

    model = DinoV2Backbone(backbone_size="small", finetune_last_n_layers=2)
    image_path = r"D:\PythonProject\RerankVPR\visualization\imgs\output_imgs\tokyo\00010.jpg"
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    x = transform(image).unsqueeze(0)
     # 获取注意力图和特征
    attention_maps, cls_token, patch_tokens = model(x)
    
    print("Attention maps shape:", attention_maps.shape)
    print("Cls token shape:", cls_token.shape)
    print("Patch tokens shape:", patch_tokens.shape)
    
    # # 可视化第一张图片的注意力图
    # visualize_attention(
    #     x[0].detach().cpu(),
    #     attention_maps[0].detach().cpu(),
    #     save_path='attention_overlay.png',
    #     alpha=0.6
    # )


