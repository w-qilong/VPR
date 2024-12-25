import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def _prepare_figure_layout(num_heads: int, heads_per_row: int, head_size: float = 3.0, 
                          dpi: int = 100) -> tuple[plt.Figure, list, int, int]:
    """准备图表布局
    
    Args:
        head_size: 每个注意力头显示的大小（英寸），默认3.0
    """
    num_rows = int(np.ceil(num_heads / heads_per_row))
    num_cols = min(heads_per_row, num_heads)
    
    # 根据head数量和每个head的大小自动计算figure尺寸
    fig_width = num_cols * head_size
    fig_height = num_rows * head_size
    
    plt.rcParams['font.family'] = 'Arial'
    fig = plt.figure(figsize=(fig_width, fig_height), dpi=dpi)
    
    # 创建子图网格
    gs = fig.add_gridspec(num_rows, num_cols)
    axes = [[fig.add_subplot(gs[i, j]) for j in range(num_cols)] for i in range(num_rows)]
    
    return fig, axes, num_rows, num_cols

def _process_attention_map(attention_tensor: torch.Tensor, original_size: tuple[int, int], 
                         original_img: np.ndarray, alpha: float, cmap: str) -> np.ndarray:
    """处理单个注意力图"""
    # 调整注意力图大小并转换为numpy数组
    attention = F.interpolate(
        attention_tensor.unsqueeze(0).unsqueeze(0),
        size=original_size[::-1],
        mode='bilinear',
        align_corners=False
    ).squeeze().cpu().numpy()
    
    # 归一化并创建热力图
    attention_norm = (attention - attention.min()) / (attention.max() - attention.min())
    attention_rgb = plt.cm.get_cmap(cmap)(attention_norm)[:, :, :3]
    
    # 叠加图像
    return np.clip((1 - alpha) * original_img + alpha * attention_rgb, 0, 1)

def visualize_multi_head_attention(attentions: torch.Tensor, 
                       original_size: tuple[int, int], 
                       image_path: str, 
                       save_path: str = None, 
                       alpha: float = 0.6,
                       heads_per_row: int = 6, 
                       head_size: float = 3.0,
                       wspace: float = 0.1,
                       dpi: int = 100,
                       cmap: str = 'jet',
                       font_family: str = 'Arial',
                       title_fontsize: int = None):
    """
    将注意力图可视化并叠加到原始图像上
    
    新增参数:
        font_family: 字体名称，默认为'Arial'
        title_fontsize: 标题字体大小，如果为None则自动计算
    """
    # 设置字体
    plt.rcParams['font.family'] = font_family
    
    # 准备原始图像
    original_img = np.array(Image.open(image_path).convert('RGB').resize(original_size)) / 255.0
    
    # 准备图表布局
    fig, axes, num_rows, num_cols = _prepare_figure_layout(
        attentions.shape[1], heads_per_row, head_size, dpi
    )
    
    # 调整子图间距
    plt.subplots_adjust(wspace=wspace, hspace=wspace)
    
    # 计算标题字体大小（如果未指定）
    if title_fontsize is None:
        title_fontsize = max(8, min(16, head_size * 3))
    
    # 处理每个注意力头
    for idx in range(attentions.shape[1]):
        row, col = idx // heads_per_row, idx % heads_per_row
        ax = axes[row][col]
        
        # 处理并显示注意力图
        overlay = _process_attention_map(attentions[0, idx], original_size, original_img, alpha, cmap)
        ax.imshow(overlay)
        ax.set_title(f'Head {idx + 1}', fontsize=title_fontsize)
        ax.axis('off')
    
    # 隐藏多余的子图
    for idx in range(attentions.shape[1], num_rows * num_cols):
        axes[idx // heads_per_row][idx % heads_per_row].set_visible(False)
    
    # 保存或显示结果
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', pad_inches=0.1)
        plt.close()
    else:
        plt.show()


def visualize_saliency_maps(
    image: torch.Tensor, 
    saliency_map: torch.Tensor, 
    alpha: float = 0.5,
    cmap: str = 'jet',
    save_path: str = None,
    figsize: tuple = (6, 6),
    sigma: float = 1.0
) -> None:
    """
    将显著性图叠加显示在原始图像上
    
    Args:
        image: 原始图像张量，形状为 [C x H x W] 或 [B x C x H x W]
        saliency_map: 显著性图张量，形状为 [t-1]
        alpha: 显著性图的透明度，范围 [0, 1]，默认为 0.5
        cmap: 热力图的颜色映射，默认为 'jet'
        save_path: 保存图像的路径，如果为None则直接显示
        figsize: 图像大小，默认为 (6, 6)
        sigma: 高斯模糊的sigma参数
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # 确保输入是CPU上的numpy数组
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(saliency_map):
        saliency_map = saliency_map.cpu().numpy()
    
    # 处理batch维度
    if len(image.shape) == 4:
        image = image[0]
    
    # 转换图像格式 [C x H x W] -> [H x W x C]
    image = np.transpose(image, (1, 2, 0))
    
    # 归一化图像到[0,1]范围
    image = (image - image.min()) / (image.max() - image.min())
    
    # 重塑显著性图到图像大小
    h, w = image.shape[:2]
    patch_size = int(np.sqrt(len(saliency_map)))
    saliency_map = saliency_map.reshape(patch_size, patch_size)
    
    # 添加高斯模糊
    saliency_map = gaussian_filter(saliency_map, sigma=sigma)
    
    # 使用更高质量的插值方法
    saliency_map = Image.fromarray(
        (saliency_map * 255).astype(np.uint8)
    ).resize((w, h), Image.Resampling.LANCZOS)
    saliency_map = np.array(saliency_map) / 255.0

    # 创建图像
    plt.figure(figsize=figsize)
    
    # 显示原始图像和叠加的显著性图
    plt.imshow(image)
    plt.imshow(saliency_map, cmap=cmap, alpha=alpha)
    plt.axis('off')
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
    else:
        plt.show()

# 用示例
def main():
    """
    主函数：展示如何使用上述函数
    """
    
    # 设置图像路径
    image_path = r"/media/cartolab3/DataDisk/wuqilong_file/Projects/RerenkVPR/tmp_imgs/00266.jpg"
    
    # 获取注意力图
    attentions, w, h = torch.randn(1, 10, 16, 16), 224, 224
    
    # 测试
    visualize_multi_head_attention(attentions, (w, h), image_path, 
                       save_path=None,
                       alpha=0.6,
                       heads_per_row=6,
                       head_size=3.0,
                       wspace=0.1)  # 增加间距

if __name__ == "__main__":
    main()
