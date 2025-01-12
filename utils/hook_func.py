from PIL import Image
from typing import Dict, List, Tuple
import torch
from torch.nn import functional as F
import os
import einops as ein
import torchvision.transforms as T

import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path

import warnings

XFORMERS_ENABLED = os.environ.get("XFORMERS_DISABLED") is None
try:
    if XFORMERS_ENABLED:
        XFORMERS_AVAILABLE = True
        warnings.warn("xFormers is available (Block)")
    else:
        warnings.warn("xFormers is disabled (Block)")
        raise ImportError
except ImportError:
    XFORMERS_AVAILABLE = False
    warnings.warn("xFormers is not available (Block)")


def get_layers_output(layers):
    """
    获取多个层的输出特征

    Args:
        layers: 需要获取输出的网络层列表

    Returns:
        dict: 包含每层输出的字典，格式为 {layer: [outputs]}
    """
    outputs = {layer: [] for layer in layers}

    def hook(module, input, output, layer_name):
        """
        钩子函数，用于捕获层的输出
        """
        outputs[layer_name].append(output)

    # 为每一层注册前向传播钩子
    for layer in layers:
        layer.register_forward_hook(
            lambda module, input, output, layer_name=layer: hook(
                module, input, output, layer_name
            )
        )

    return outputs


def get_hook(model, facet: str, feats: List):
    """
    生成特定类型的钩子函数

    Args:
        model: 模型实例
        facet: 特征类型，可选值：
            - 'attn': 注意力权重矩阵
            - 'token': token特征
            - 'query': 查询向量
            - 'key': 键向量
            - 'value': 值向量
        feats: 存储提取特征的列表

    Returns:
        callable: 钩子函数
    """
    if facet in ["attn", "token"]:
        if XFORMERS_AVAILABLE and facet == "attn":

            def _hook(module, input, output):
                # 提取注意力权重
                input = input[0]
                B, N, C = input.shape
                scale = module.scale
                # 重塑 QKV 矩阵
                qkv = (
                    module.qkv(input)
                    .reshape(B, N, 3, module.num_heads, C // module.num_heads)
                    .permute(2, 0, 3, 1, 4)
                )
                q, k, v = qkv[0] * scale, qkv[1], qkv[2]
                # 计算注意力分数
                attn = q @ k.transpose(-2, -1)
                feats.append(attn)

            return _hook
        else:

            def _hook(model, input, output):
                feats.append(output)

            return _hook

    # 确定特征类型对应的索引
    facet_indices = {"query": 0, "key": 1, "value": 2}

    if facet not in facet_indices:
        raise TypeError(f"不支持的特征类型: {facet}")

    facet_idx = facet_indices[facet]

    def _inner_hook(module, input, output):
        """
        内部钩子函数，用于提取 QKV 向量
        """
        input = input[0]
        B, N, C = input.shape
        # 提取并重塑 QKV 矩阵
        qkv = (
            module.qkv(input)
            .reshape(B, N, 3, module.num_heads, C // module.num_heads)
            .permute(2, 0, 3, 1, 4)
        )
        feats.append(qkv[facet_idx])  # 形状: [B x h x t x d]

    return _inner_hook


def register_hooks(
    model, layers_and_facet: Dict, hook_handlers: List, feats: List
) -> None:
    """
    注册钩子函数以提取模型特定层的特征。

    Args:
        model: 需要提取特征的模型实例
        layers_and_facet: 需要提取特征的层索引和特征类型字典，例如 {0: 'token', 1: 'attn'}
        hook_handlers: 存储钩子函数句柄的列表，用于后续移除
        feats: 存储提取特征的列表

    Raises:
        TypeError: 当指定了不支持的特征类型时抛出
    """
    # 定义支持的特征类型
    supported_facets = {"token", "attn", "key", "query", "value"}
   

    # 遍历模型的每个块
    for block_idx, block in enumerate(model.blocks):
        if block_idx in layers_and_facet:

            # 根据不同的特征类型注册相应的钩子
            facet = layers_and_facet[block_idx]
            if facet not in supported_facets:   
                raise TypeError(f"不支持的特征类型: {facet}")
            
            if facet == "token":
                # 注册token级特征的钩子
                hook_handlers.append(
                    block.register_forward_hook(get_hook(model, facet, feats))
                )
            elif facet == "attn":
                # 注册注意力特征的钩子，需要考虑xFormers的可用性
                if XFORMERS_AVAILABLE:
                    hook_handlers.append(
                        block.attn.register_forward_hook(get_hook(model, facet, feats))
                    )
                else:
                    hook_handlers.append(
                        block.attn.attn_drop.register_forward_hook(
                            get_hook(model, facet, feats)
                        )
                    )
            else:  # key, query, value
                # 注册QKV相关特征的钩子
                hook_handlers.append(
                    block.attn.register_forward_hook(get_hook(model, facet, feats))
                )


def unregister_hooks(hook_handlers: List) -> None:
    """
    移除所有已注册的钩子函数。

    Args:
        hook_handlers: 包含钩子函数句柄的列表

    Note:
        - 此函数会清空hook_handlers列表
        - 建议在特征提取完成后立即调用此函数，以防止内存泄漏
    """
    for handle in hook_handlers:
        handle.remove()
    hook_handlers.clear()  # 使用clear()替代重新赋值，更高效


def extract_features(
    model, batch: torch.Tensor, layers_and_facet: Dict
) -> List[torch.Tensor]:
    """
    从模型的指定层中提取特定类型的特征。

    Args:
        model: 要提取特征的模型实例
        batch: 输入数据张量，形状为 [B, C, H, W]
            B: 批次大小
            C: 通道数
            H: 高度
            W: 宽度
        layers: 要提取特征的层索引列表
            - ViT-B/16模型：取值范围 0-11
            - ViT-L/16模型：取值范围 0-23
        facet: 特征类型，默认为'key'
            可选值：'key', 'query', 'value', 'token', 'attn'

    Returns:
        List[torch.Tensor]: 提取的特征列表
        特征张量的形状取决于facet参数：
            - key/query/value: [B, h, t, d] (批次, 头数, 序列长度, 维度)
            - attn: [B, h, t, t] (批次, 头数, 序列长度, 序列长度)
            - token: [B, t, d] (批次, 序列长度, 维度)

    Example:
        >>> batch = torch.randn(32, 3, 224, 224)  # 32张224x224的RGB图像
        >>> layers = [0, 1, 2]  # 提取前三层的特征
        >>> features = extract_features(model, batch, layers, facet='key')
    """
    # 验证输入数据的形状
    B, C, H, W = batch.shape
    assert C == 3, f"输入数据应该是RGB图像，但收到了{C}个通道"

    # 初始化特征列表和钩子句柄列表
    feats = []
    hook_handlers = []

    try:
        # 注册钩子
        register_hooks(model, layers_and_facet, hook_handlers, feats)
        # 前向传播
        _ = model(batch)
    finally:
        # 确保钩子被移除，即使发生异常
        unregister_hooks(hook_handlers)

    return feats


def extract_saliency_maps(
    model, layers_and_facet: Dict, batch: torch.Tensor
) -> torch.Tensor:
    """
    提取显著性图（Saliency Maps）

    通过提取模型最后一层CLS token的注意力头的平均值来生成显著性图。
    所有值都被归一化到0到1之间。

    Args:
        model: 要提取特征的模型实例
        layers_and_facet: 要提取特征的层索引和特征类型字典 {0: 'attn'},提取特定层的注意力特征
        batch: 输入数据批次，形状为 BxCxHxW
            B: 批次大小
            C: 通道数
            H: 图像高度
            W: 图像宽度

    Returns:
        torch.Tensor: 显著性图张量，形状为 Bx(t-1)
            B: 批次大小
            t-1: 序列长度减1（不包含CLS token）
    """
    # 提取注意力特征
    feats = extract_features(model, batch, layers_and_facet)

    # 获取最后一层的特征 [B x h x t x t]
    curr_feats = feats[0]

    # 提取CLS token的注意力图并计算平均值
    # 选择 [:,:,0,1:] 表示：
    # - 所有批次样本 [:]
    # - 所有注意力头 [:]
    # - CLS token的注意力 [0]
    # - 除CLS token外的所有token [1:]
    cls_attn_map = curr_feats[:, :, 0, 1:].mean(dim=1)  # [B x (t-1)]

    # 对每个样本进行归一化
    temp_mins = cls_attn_map.min(dim=1)[0]  # [B]
    temp_maxs = cls_attn_map.max(dim=1)[0]  # [B]
    cls_attn_maps = (cls_attn_map - temp_mins.unsqueeze(1)) / (
        temp_maxs - temp_mins
    ).unsqueeze(1)

    return cls_attn_maps


def log_bin(
    x: torch.Tensor, num_patches: Tuple[int, int], hierarchy: int = 2
) -> torch.Tensor:
    """
    创建对特征进行对数分箱的描述符。
    主要创新点：
    1.对数空间分箱(Log-space Binning)：
        借鉴了传统计算机视觉中的 SIFT (Scale-Invariant Feature Transform) 描述符的思想
        通过多尺度特征聚合提高特征的判别性
        层级化特征提取：
    2.层级化特征提取：
        使用3^k大小的核进行平均池化
        每个层级捕获不同尺度的上下文信息
        类似于图像金字塔的概念，但在特征空间进行操作
    3.这种方法的优势：
        提高了特征的鲁棒性
        更好地处理尺度变化
        保留了局部和全局信息
        适合进行密集特征匹配

    Args:
        x: 特征张量，形状为 [B, h, t, d]
            B: 批次大小
            h: 注意力头数
            t: token序列长度
            d: 特征维度
        num_patches: 图像patch的数量，格式为 (height, width)
        hierarchy: 分箱层级数，默认为2
            每个层级使用 3^k 大小的核进行平均池化

    Returns:
        torch.Tensor: 分箱后的描述符，形状为 [B, 1, t-1, d*h*num_bins]
            其中 num_bins = 1 + 8 * hierarchy
    """
    B = x.shape[0]
    num_bins = 1 + 8 * hierarchy

    # 重排张量维度并展平
    bin_x = x.permute(0, 2, 3, 1).flatten(start_dim=-2, end_dim=-1)  # [B, t-1, d*h]
    bin_x = bin_x.permute(0, 2, 1)  # [B, d*h, t-1]
    bin_x = bin_x.reshape(
        B, bin_x.shape[1], num_patches[0], num_patches[1]
    )  # [B, d*h, H, W]
    sub_desc_dim = bin_x.shape[1]

    # 计算不同尺度的平均池化特征
    avg_pools = []
    for k in range(hierarchy):
        win_size = 3**k
        avg_pool = torch.nn.AvgPool2d(
            win_size, stride=1, padding=win_size // 2, count_include_pad=False
        )
        avg_pools.append(avg_pool(bin_x))

    # 初始化输出张量
    bin_x = torch.zeros(
        (B, sub_desc_dim * num_bins, num_patches[0], num_patches[1]), device=x.device
    )

    # 为每个空间位置填充所有尺度的分箱
    for y in range(num_patches[0]):
        for x in range(num_patches[1]):
            part_idx = 0
            for k in range(hierarchy):
                kernel_size = 3**k
                # 遍历当前核大小覆盖的所有位置
                for i in range(y - kernel_size, y + kernel_size + 1, kernel_size):
                    for j in range(x - kernel_size, x + kernel_size + 1, kernel_size):
                        # 跳过非第一层的中心点
                        if i == y and j == x and k != 0:
                            continue

                        # 处理边界情况
                        if 0 <= i < num_patches[0] and 0 <= j < num_patches[1]:
                            feat = avg_pools[k][:, :, i, j]
                        else:
                            # 边界填充：使用最近的有效位置
                            temp_i = max(0, min(i, num_patches[0] - 1))
                            temp_j = max(0, min(j, num_patches[1] - 1))
                            feat = avg_pools[k][:, :, temp_i, temp_j]

                        # 存储特征
                        bin_x[
                            :,
                            part_idx * sub_desc_dim : (part_idx + 1) * sub_desc_dim,
                            y,
                            x,
                        ] = feat
                        part_idx += 1

    # 重排输出张量维度
    bin_x = bin_x.flatten(start_dim=-2, end_dim=-1)  # [B, d*h*num_bins, H*W]
    bin_x = bin_x.permute(0, 2, 1).unsqueeze(dim=1)  # [B, 1, H*W, d*h*num_bins]

    return bin_x


def extract_descriptors(
    model,
    batch: torch.Tensor,
    layers_and_facet: Dict={23: 'value'},
    bin: bool = False,
    include_cls: bool = False,
    num_patches: Tuple[int, int] = (14, 14),
    hierarchy: int = 2,
) -> torch.Tensor:
    """
    从模型中提取特征描述符。

    Args:
        model: 要提取特征的模型实例
        batch: 输入数据批次，形状为 [B, C, H, W]
            B: 批次大小
            C: 通道数（通常为3，表示RGB图像）
            H: 图像高度
            W: 图像宽度
        layers_and_facet: 要提取特征的层索引和特征类型字典
            - ViT-B/16模型：取值范围 [0, 11]
            - ViT-L/16模型：取值范围 [0, 23]
        bin: 是否对描述符应用对数分箱，默认为False
        include_cls: 是否包含CLS token，默认为False
            注意：当bin=True时，include_cls必须为False
        num_patches: patch的数量，格式为(height, width)，默认为(14, 14)
        hierarchy: 分箱层级数，默认为2
            每个层级使用 3^k 大小的核进行平均池化

    Returns:
        torch.Tensor: 描述符张量
            - 当bin=False时，形状为 [B, 1, t, d*h]
            - 当bin=True时，形状为 [B, 1, t-1, d*h*num_bins]
            其中：
                t: token序列长度
                d: 特征维度
                h: 注意力头数
                num_bins: 分箱数量

    Raises:
        AssertionError:
            - 当facet不是支持的类型时
            - 当bin=True且include_cls=True时
    """
    # 验证输入参数
    supported_facets = {"key", "query", "value", "token"}
    facet = list(layers_and_facet.values())[0]
    if facet not in supported_facets:
        raise ValueError(f"不支持的特征类型: {facet}，可选值: {supported_facets}")

    if bin and include_cls:
        raise ValueError("bin=True 和 include_cls=True 不能同时使用")

    # 提取特征并获取第一层的输出
    feats = extract_features(model, batch, layers_and_facet)
    x = feats[0]  # shape: [B, h, t, d] 或 [B, t, d]（对于token）
    
    # 处理token特征，统一维度格式
    if facet == "token":
        x = x.unsqueeze(1)  # 添加头维度 [B, 1, t, d]

    # 处理CLS token
    if not include_cls:
        x = x[:, :, 1:]  # 移除CLS token

    # 生成最终描述符
    if not bin:
        # 重排维度并合并最后两个维度
        desc = x.permute(0, 2, 3, 1)  # [B, t, d, h]
        desc = desc.flatten(start_dim=-2)  # [B, t, d*h]
        desc = desc.unsqueeze(1)  # [B, 1, t, d*h]
    else:
        desc = log_bin(x, num_patches, hierarchy=hierarchy)  # [B, 1, t-1, d*h*num_bins]

    return desc

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """计算两组向量之间所有可能对的余弦相似度。
    使用矩阵乘法实现快速计算。

    Args:
        x: 描述符张量，形状为 [B, 1, t_x, d]
        y: 描述符张量，形状为 [B, 1, t_y, d]

    Returns:
        torch.Tensor: 余弦相似度矩阵，形状为 [B, 1, t_x, t_y]
    """
    
    # 计算 L2 范数
    x_norm = torch.norm(x, p=2, dim=-1, keepdim=True)  # [B, 1, t_x, 1]
    y_norm = torch.norm(y, p=2, dim=-1, keepdim=True)  # [B, 1, t_y, 1]

    # 归一化向量
    x_normalized = x / (x_norm + 1e-8)  # [B, 1, t_x, d]
    y_normalized = y / (y_norm + 1e-8)  # [B, 1, t_y, d]

    # 使用矩阵乘法计算相似度
    # 调整维度以进行批量矩阵乘法
    sim = torch.matmul(x_normalized, y_normalized.transpose(-2, -1))  # [B, 1, t_x, t_y]
    
    return sim

def find_correspondences(
    model,
    transform,
    image_path1: str,
    image_path2: str,
    facet_layer_and_facet: Dict = {23: 'value'},
    saliency_map_layer_and_facet: Dict = {23: 'attn'},
    bin: bool = True,
    include_cls: bool = False,
    thresh: float = 0.05,
    num_pairs: int = 10,
    hierarchy: int = 2,
) -> Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:
    """
    在两张图像之间找到对应点对。

    Args:
        model: 用于特征提取的模型
        transform: 图像预处理转换
        image_path1: 第一张图像的路径
        image_path2: 第二张图像的路径
        facet_layer_and_facet: 要提取特征的层索引和特征类型字典
        saliency_map_layer_and_facet: 要提取显著性图的层索引和特征类型字典
        bin: 是否使用对数分箱，默认为True
        include_cls: 是否包含CLS token，默认为False
        thresh: 显著性图阈值，默认为0.05
        num_pairs: 返回的对应点对数量，默认为10
        hierarchy: 分箱层级数，默认为2
            每个层级使用 3^k 大小的核进行平均池化

    Returns:
        Tuple[List[Tuple[float, float]], List[Tuple[float, float]], Image.Image, Image.Image]:
            - points1: 第一张图像中的对应点坐标列表
            - points2: 第二张图像中的对应点坐标列表
            - image1: 第一张PIL图像
            - image2: 第二张PIL图像
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # 1. 图像加载和预处理
    image1 = Image.open(image_path1).convert("RGB")
    image2 = Image.open(image_path2).convert("RGB")
    
    image_batch1 = transform(image1).unsqueeze(0).to(device)
    image_batch2 = transform(image2).unsqueeze(0).to(device)
    
    load_size = image_batch1.shape[-1]
    num_patches = (int(load_size / 14), int(load_size / 14))

    image1=image1.resize((load_size, load_size))
    image2=image2.resize((load_size, load_size))

    # 2. 特征提取
    descriptors1 = extract_descriptors(
        model, image_batch1, facet_layer_and_facet, bin,
        include_cls=include_cls, num_patches=num_patches, hierarchy=hierarchy
    ) # shape: [1, 1, 1600, d*h*num_bins]
    descriptors2 = extract_descriptors(
        model, image_batch2, facet_layer_and_facet, bin,
        include_cls=include_cls, num_patches=num_patches, hierarchy=hierarchy
    ) # shape: [1, 1, 256, d*h*num_bins]

    print(descriptors1.shape, descriptors2.shape)

    # 3. 提取并处理显著性图
    saliency_map1 = extract_saliency_maps(model, saliency_map_layer_and_facet, image_batch1)[0].to(device)
    saliency_map2 = extract_saliency_maps(model, saliency_map_layer_and_facet, image_batch2)[0].to(device)
    print(saliency_map1.shape, saliency_map2.shape)

    fg_mask1 = saliency_map1 > thresh
    fg_mask2 = saliency_map2 > thresh

    # 4. 计算描述符之间的相似度
    similarities = chunk_cosine_sim(descriptors1, descriptors2)

    # 5. 寻找最佳匹配点对（Best Buddies）
    print(num_patches[0] ,num_patches[1])
    image_idxs = torch.arange(num_patches[0] * num_patches[1], device=device)
    sim_1, nn_1 = torch.max(similarities, dim=-1)  # 图像2中与图像1最相似的点
    sim_2, nn_2 = torch.max(similarities, dim=-2)  # 图像1中与图像2最相似的点
    sim_1, nn_1 = sim_1[0, 0], nn_1[0, 0]
    sim_2, nn_2 = sim_2[0, 0], nn_2[0, 0]
    bbs_mask = nn_2[nn_1] == image_idxs  # 互为最佳匹配的点对

    print(f"互为最佳匹配的点对数量: {bbs_mask.int().sum().item()}")

    # 6. 基于显著性掩码过滤匹配点
    fg_mask2_new_coors = nn_2[fg_mask2]
    fg_mask2_mask_new_coors = torch.zeros(
        num_patches[0] * num_patches[1], 
        dtype=torch.bool, 
        device=device
    )
    fg_mask2_mask_new_coors[fg_mask2_new_coors] = True
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask1)
    bbs_mask = torch.bitwise_and(bbs_mask, fg_mask2_mask_new_coors)

    print(f"基于显著性掩码过滤后的点对数量: {bbs_mask.int().sum().item()}")

    # 7. 使用K-means聚类选择高质量的对应点对
    bb_descs1 = descriptors1[0, 0, bbs_mask, :].detach().cpu().numpy()
    bb_descs2 = descriptors2[0, 0, nn_1[bbs_mask], :].detach().cpu().numpy()
    all_keys_together = np.concatenate((bb_descs1, bb_descs2), axis=1)
    
    # 确定聚类数量并归一化特征
    n_clusters = min(num_pairs, len(all_keys_together))
    length = np.sqrt((all_keys_together ** 2).sum(axis=1))[:, None]
    normalized = all_keys_together / length
    
    # 执行K-means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=0).fit(normalized)
    bb_topk_sims = np.full((n_clusters), -np.inf)
    bb_indices_to_show = np.full((n_clusters), -np.inf)

    # 8. 基于显著性值对每个聚类中的点对进行排序
    bb_cls_attn = (
        saliency_map1[bbs_mask] + 
        saliency_map2[nn_1[bbs_mask]]
    ) / 2
    
    for k in range(n_clusters):
        for i, (label, rank) in enumerate(zip(kmeans.labels_, bb_cls_attn)):
            if rank > bb_topk_sims[label]:
                bb_topk_sims[label] = rank
                bb_indices_to_show[label] = i

    # 9. 计算最终的对应点坐标
    indices_to_show = torch.nonzero(bbs_mask, as_tuple=False).squeeze(dim=1)[
        bb_indices_to_show
    ]
    img1_indices_to_show = torch.arange(num_patches[0] * num_patches[1], device=device)[
        indices_to_show
    ]
    img2_indices_to_show = nn_1[indices_to_show]

    # 转换为图像坐标系
    img1_y_to_show = (img1_indices_to_show / num_patches[1]).cpu().numpy()
    img1_x_to_show = (img1_indices_to_show % num_patches[1]).cpu().numpy()
    img2_y_to_show = (img2_indices_to_show / num_patches[1]).cpu().numpy()
    img2_x_to_show = (img2_indices_to_show % num_patches[1]).cpu().numpy()

    # 10. 生成最终的对应点列表
    points1, points2 = [], []
    stride = (14, 14)
    patch_size = 14
    
    for y1, x1, y2, x2 in zip(img1_y_to_show, img1_x_to_show, img2_y_to_show, img2_x_to_show):
        # 计算patch中心点坐标
        x1_show = (int(x1) - 1) * stride[1] + stride[1] + patch_size // 2
        y1_show = (int(y1) - 1) * stride[0] + stride[0] + patch_size // 2
        x2_show = (int(x2) - 1) * stride[1] + stride[1] + patch_size // 2
        y2_show = (int(y2) - 1) * stride[0] + stride[0] + patch_size // 2
        
        points1.append((y1_show, x1_show))
        points2.append((y2_show, x2_show))

    return points1, points2, image1, image2


def draw_correspondences(
    points1: List[Tuple[float, float]], 
    points2: List[Tuple[float, float]],
    image1: Image.Image, 
    image2: Image.Image
) -> Tuple[plt.Figure, plt.Figure]:
    """
    在两张图像上绘制对应点的可视化结果。

    Args:
        points1: 图像1中的坐标点列表，每个元素为(y, x)坐标
        points2: 图像2中的坐标点列表，每个元素为(y, x)坐标
        image1: 第一张PIL图像
        image2: 第二张PIL图像

    Returns:
        Tuple[plt.Figure, plt.Figure]: 包含标记点的两个图像figure对象

    Notes:
        - 每个对应点用相同颜色标记
        - 对于点数<=15使用固定颜色列表，>15使用tab10色彩映射
        - 每个点用两个同心圆标记:
            - 外圆: 半透明，半径8
            - 内圆: 不透明，半径1
    """
    # 验证输入点对数量是否匹配
    assert len(points1) == len(points2), f"对应点数量不匹配: {len(points1)} != {len(points2)}"
    
    # 创建两个图像的绘图对象
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    ax1.axis('off')
    ax2.axis('off')
    ax1.imshow(image1)
    ax2.imshow(image2)

    # 根据点数选择合适的颜色映射
    num_points = len(points1)
    if num_points > 15:
        cmap = plt.get_cmap('tab10')
    else:
        # 使用预定义的固定颜色列表
        fixed_colors = [
            "red", "yellow", "blue", "lime", "magenta", 
            "indigo", "orange", "cyan", "darkgreen", "maroon",
            "black", "white", "chocolate", "gray", "blueviolet"
        ]
        cmap = ListedColormap(fixed_colors)
    
    # 生成颜色数组
    colors = np.array([cmap(x) for x in range(num_points)])
    
    # 定义内外圆半径
    OUTER_RADIUS, INNER_RADIUS = 8, 1

    # 绘制对应点
    for point1, point2, color in zip(points1, points2, colors):
        # 绘制图像1上的点
        y1, x1 = point1
        ax1.add_patch(plt.Circle(
            (x1, y1), 
            OUTER_RADIUS, 
            facecolor=color, 
            edgecolor='white', 
            alpha=0.5
        ))
        ax1.add_patch(plt.Circle(
            (x1, y1), 
            INNER_RADIUS, 
            facecolor=color, 
            edgecolor='white'
        ))

        # 绘制图像2上的点
        y2, x2 = point2
        ax2.add_patch(plt.Circle(
            (x2, y2), 
            OUTER_RADIUS, 
            facecolor=color, 
            edgecolor='white', 
            alpha=0.5
        ))
        ax2.add_patch(plt.Circle(
            (x2, y2), 
            INNER_RADIUS, 
            facecolor=color, 
            edgecolor='white'
        ))

    return fig1, fig2


def draw_correspondences_with_lines(
    points1: List[Tuple[float, float]], 
    points2: List[Tuple[float, float]],
    image1: Image.Image, 
    image2: Image.Image
) -> plt.Figure:
    """
    在同一张图中绘制两张图像及其对应点连线。

    Args:
        points1: 图像1中的坐标点列表，每个元素为(y, x)坐标
        points2: 图像2中的坐标点列表，每个元素为(y, x)坐标
        image1: 第一张PIL图像
        image2: 第二张PIL图像

    Returns:
        plt.Figure: 包含两张图像和对应点连线的figure对象

    Notes:
        - 两张图像横向排列
        - 所有对应点使用相同的颜色标记
        - 对应点之间的连线使用随机颜色
        - 每个点用两个同心圆标记:
            - 外圆: 半透明，半径8
            - 内圆: 不透明，半径1
    """
    # 验证输入点对数量是否匹配
    assert len(points1) == len(points2), f"对应点数量不匹配: {len(points1)} != {len(points2)}"
    
    # 创建图像绘图对象
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.axis('off')
    
    # 计算两张图像的间距和偏移量
    img_width = image1.size[0]
    gap = 10  # 图像之间的间距
    offset = img_width + gap  # 第二张图像的x轴偏移量
    
    # 显示两张图像
    ax.imshow(image1, extent=[0, img_width, 0, image1.size[1]])
    ax.imshow(image2, extent=[offset, offset + img_width, 0, image2.size[1]])
    
    # 设置点的颜色和大小
    POINT_COLOR = 'red'  # 所有点使用相同的颜色
    OUTER_RADIUS, INNER_RADIUS = 8, 1
    
    # 为每条连线生成随机颜色
    num_points = len(points1)
    np.random.seed(42)  # 设置随机种子以保持颜色一致性
    line_colors = np.random.rand(num_points, 3)  # RGB随机颜色
    
    # 绘制对应点和连线
    for point1, point2, line_color in zip(points1, points2, line_colors):
        y1, x1 = point1
        y2, x2 = point2
        
        # 调整第二张图像中点的x坐标
        x2_adjusted = x2 + offset
        
        # 绘制连接线
        ax.plot([x1, x2_adjusted], [y1, y2], 
                color=line_color, alpha=0.6, linewidth=2)
        # alpha参数控制线条的透明度，范围为0到1，0为完全透明，1为不透明
        
        # 绘制图像1上的点
        ax.add_patch(plt.Circle(
            (x1, y1), 
            OUTER_RADIUS, 
            facecolor=line_color, 
            edgecolor='white', 
            alpha=0.5
        ))
        ax.add_patch(plt.Circle(
            (x1, y1), 
            INNER_RADIUS, 
            facecolor=line_color, 
            edgecolor='white'
        ))
        
        # 绘制图像2上的点
        ax.add_patch(plt.Circle(
            (x2_adjusted, y2), 
            OUTER_RADIUS, 
            facecolor=line_color, 
            edgecolor='white', 
            alpha=0.5
        ))
        ax.add_patch(plt.Circle(
            (x2_adjusted, y2), 
            INNER_RADIUS, 
            facecolor=line_color, 
            edgecolor='white'
        ))
    
    # 调整图像显示范围
    ax.set_xlim(-10, offset + img_width + 10)
    ax.set_ylim(-10, max(image1.size[1], image2.size[1]) + 10)
    
    return fig

def compute_visual_similarity(
    source_path: str,
    target_path: str,
    model: torch.nn.Module,
    query_point: Tuple[int, int] = (280, 280),
    layer_idx: int = 22,
    image_size: int = 560,
    device: str = "cuda"
) -> Tuple[Dict[str, np.ndarray], Image.Image, Image.Image]:
    """
    计算两张图片指定位置的视觉特征相似度。
    
    Args:
        source_path: 源图像路径（查询图像）
        target_path: 目标图像路径（对比图像）
        model: 预训练的视觉Transformer模型
        query_point: 查询点坐标 (x, y)，默认(280, 280)
        layer_idx: 提取特征的transformer层索引，默认22
        image_size: 处理图像的大小，默认560
        device: 计算设备，默认"cuda"
    
    Returns:
        Tuple[Dict[str, np.ndarray], Image.Image, Image.Image]:
            - 特征相似度字典，包含'key', 'query', 'value', 'token'四种特征的相似度图
            - 源图像PIL对象
            - 目标图像PIL对象
    """
    # 1. 图像预处理
    patch_size = 14
    num_patches = image_size // patch_size
    
    transform = _create_image_transform(image_size)
    source_img, target_img, source_pil, target_pil = _load_and_preprocess_images(
        source_path, target_path, transform, device
    )
    
    # 2. 计算特征相似度
    similarity_maps = {}
    hook_outputs = []
    
    for feature_type in ["key", "query", "value", "token"]:
        # 提取特征
        source_feat, target_feat = _extract_features(
            model, source_img, target_img, feature_type, 
            layer_idx, hook_outputs
        )
        
        # 计算相似度图
        similarity_map = _compute_similarity_map(
            source_feat, target_feat,
            query_point, num_patches, image_size
        )
        similarity_maps[feature_type] = similarity_map
    
    return similarity_maps, source_pil.resize((image_size, image_size)), target_pil.resize((image_size, image_size))

def _create_image_transform(image_size: int) -> T.Compose:
    """创建图像预处理转换"""
    return T.Compose([
        T.Resize((image_size, image_size), 
                 interpolation=T.InterpolationMode.BILINEAR),
        T.ToTensor(),
        T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def _load_and_preprocess_images(
    source_path: str,
    target_path: str,
    transform: T.Compose,
    device: str
) -> Tuple[torch.Tensor, torch.Tensor, Image.Image, Image.Image]:
    """加载并预处理图像"""
    source_pil = Image.open(source_path).convert("RGB")
    target_pil = Image.open(target_path).convert("RGB")
    
    source_tensor = transform(source_pil).unsqueeze(0).to(device)
    target_tensor = transform(target_pil).unsqueeze(0).to(device)
    
    return source_tensor, target_tensor, source_pil, target_pil

def _extract_features(
    model: torch.nn.Module,
    source_img: torch.Tensor,
    target_img: torch.Tensor,
    feature_type: str,
    layer_idx: int,
    hook_outputs: List
) -> Tuple[torch.Tensor, torch.Tensor]:
    """提取特定类型的特征"""
    hook_outputs.clear()
    
    def _forward_hook(module, inputs, output):
        hook_outputs.append(output)
    
    # 注册hook
    if feature_type == "token":
        hook = model.blocks[layer_idx].register_forward_hook(_forward_hook)
    else:
        hook = model.blocks[layer_idx].attn.qkv.register_forward_hook(_forward_hook)
    
    # 提取特征
    with torch.no_grad():
        model(source_img)
        model(target_img)
        
        source_feat = hook_outputs[0][:, 1:]  # 移除CLS token
        target_feat = hook_outputs[1][:, 1:]
        
        if feature_type != "token":
            source_feat, target_feat = _split_qkv_features(
                source_feat, target_feat, feature_type
            )
        
        # 特征归一化
        source_feat = F.normalize(source_feat, dim=-1)
        target_feat = F.normalize(target_feat, dim=-1)
    
    hook.remove()
    return source_feat, target_feat

def _split_qkv_features(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    feature_type: str
) -> Tuple[torch.Tensor, torch.Tensor]:
    """分离QKV特征"""
    feature_dim = source_feat.shape[2] // 3
    feature_indices = {
        "query": (0, feature_dim),
        "key": (feature_dim, 2*feature_dim),
        "value": (2*feature_dim, 3*feature_dim)
    }
    start_idx, end_idx = feature_indices[feature_type]
    
    return (
        source_feat[:, :, start_idx:end_idx],
        target_feat[:, :, start_idx:end_idx]
    )

def _compute_similarity_map(
    source_feat: torch.Tensor,
    target_feat: torch.Tensor,
    query_point: Tuple[int, int],
    num_patches: int,
    image_size: int
) -> np.ndarray:
    """计算特征相似度图"""
    # 重排特征维度并插值到目标大小
    source_feat = ein.rearrange(
        source_feat[0],
        "(p_h p_w) d -> d p_h p_w",
        p_h=num_patches,
        p_w=num_patches
    )[None]
    
    target_feat = ein.rearrange(
        target_feat[0],
        "(p_h p_w) d -> d p_h p_w",
        p_h=num_patches,
        p_w=num_patches
    )[None]
    
    source_feat = F.interpolate(
        source_feat,
        size=(image_size, image_size),
        mode='nearest'
    )
    target_feat = F.interpolate(
        target_feat,
        size=(image_size, image_size),
        mode='nearest'
    )
    
    # 提取查询点特征并计算相似度
    query_feat = source_feat[[0], ..., query_point[1], query_point[0]]
    query_feat = ein.repeat(
        query_feat,
        "1 d -> 1 d h w",
        h=image_size,
        w=image_size
    )
    
    similarity = F.cosine_similarity(target_feat, query_feat, dim=1).detach().cpu().numpy()
    return ein.rearrange(
        similarity,
        "1 h w -> h w 1"
    )


def create_attention_similarity_plot(
    source_image: Image.Image,
    target_image: Image.Image,
    attention_maps: dict,
    source_point: tuple,
    max_positions: dict,
    fig_size: tuple = (36, 8),
    dpi: int = 500
) -> None:
    """可视化注意力相似度图和关键点位置
    
    Args:
        source_image: 源图像PIL对象
        target_image: 目标图像PIL对象
        attention_maps: 包含key/query/value/token的注意力图字典
        source_point: 源图像上的查询点坐标 (x,y)
        max_positions: 包含各注意力图最大值位置的字典
        fig_size: 图像大小
        dpi: 图像分辨率
    """
    
    # 设置绘图样式
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'font.size': 24
    })
    
    # 标记点样式
    marker_props = {
        "ms": 20,           # marker size
        "mew": 2,          # marker edge width
        "mec": 'white',    # marker edge color
        "alpha": 0.5       # transparency
    }
    
    # 注意力类型对应的颜色
    attention_colors = {
        "key": "tab:pink",
        "query": "tab:brown", 
        "value": "tab:orange",
        "token": "tab:purple",
        "prompt": "red"
    }
    
    # 创建画布和网格
    fig = plt.figure(figsize=fig_size, dpi=dpi)
    gs = fig.add_gridspec(1, 6)
    
    # 绘制源图像
    ax_source = fig.add_subplot(gs[0, 0])
    ax_source.imshow(source_image)
    ax_source.plot(*source_point, 'o', c=attention_colors["prompt"], **marker_props)
    ax_source.axis('off')
    ax_source.set_title("Source Image", pad=10)
    
    # 绘制目标图像和最大响应点
    ax_target = fig.add_subplot(gs[0, 1])
    ax_target.imshow(target_image)
    for attn_type in ["key", "query", "value", "token"]:
        pos = max_positions[attn_type]
        ax_target.plot(pos[1], pos[0], 'o', 
                      label=attn_type,
                      c=attention_colors[attn_type], 
                      **marker_props)
    ax_target.axis('off')
    ax_target.set_title("Target Image", pad=10)
    
    # 绘制注意力图
    def normalize_attention(x):
        """归一化注意力图到0-255范围"""
        return (((x/2.0) + 0.5) * 255).astype(np.uint8)
    
    for idx, attn_type in enumerate(["key", "query", "value", "token"]):
        ax = fig.add_subplot(gs[0, idx+2])
        ax.set_title(attn_type.capitalize(), pad=10)
        ax.imshow(normalize_attention(attention_maps[attn_type]), 
                 vmin=0, vmax=255, cmap="jet")
        ax.axis('off')
    
    # 添加图例
    fig.legend(loc="lower center", ncol=4,
              bbox_to_anchor=(0.01, 0.01, 0.3, 0.08),
              mode="expand", frameon=False, fontsize=30)
    
    plt.tight_layout(pad=1.0, w_pad=0.8, h_pad=0.8)
    
    return fig









