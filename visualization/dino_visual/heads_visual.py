import torch.nn.functional as F

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

    image_path = r"D:\PythonProject\RerankVPR\visualization\imgs\output_imgs\tokyo\00010.jpg"
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    x = transform(image).unsqueeze(0)

    # 获取注意力图和特征
    attention_maps, cls_token, patch_tokens, (q, k, v) = model(x)

    # 可视化第一张图片的注意力图
    visualize_attention(
        x[0].detach().cpu(),
        attention_maps[0].detach().cpu(),
        save_path='attention_overlay.png',
        alpha=0.6
    )
