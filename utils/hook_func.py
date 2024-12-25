from PIL import Image
from typing import List
import torch
import os
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
            lambda module, input, output, layer_name=layer: 
            hook(module, input, output, layer_name)
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
    if facet in ['attn', 'token']:
        if XFORMERS_AVAILABLE and facet == 'attn':
            def _hook(module, input, output):
                # 提取注意力权重
                input = input[0]
                B, N, C = input.shape
                scale = module.scale
                # 重塑 QKV 矩阵
                qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, 
                    C // module.num_heads).permute(2, 0, 3, 1, 4)
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
    facet_indices = {
        'query': 0,
        'key': 1,
        'value': 2
    }
    
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
        qkv = module.qkv(input).reshape(B, N, 3, module.num_heads, 
            C // module.num_heads).permute(2, 0, 3, 1, 4)
        feats.append(qkv[facet_idx])  # 形状: [B x h x t x d]
    
    return _inner_hook

def register_hooks(model, layers: List[int], facet: str, hook_handlers: List, feats: List) -> None:
    """
    注册钩子函数以提取模型特定层的特征。
    
    Args:
        model: 需要提取特征的模型实例
        layers: 需要提取特征的层索引列表，例如 [0, 1, 2]
        facet: 特征类型，可选值：
            - 'attn': 注意力权重矩阵
            - 'token': token特征
            - 'query': 查询向量
            - 'key': 键向量
            - 'value': 值向量
        hook_handlers: 存储钩子函数句柄的列表，用于后续移除
        feats: 存储提取特征的列表
    
    Raises:
        TypeError: 当指定了不支持的特征类型时抛出
    """
    # 定义支持的特征类型
    supported_facets = {'token', 'attn', 'key', 'query', 'value'}
    if facet not in supported_facets:
        raise TypeError(f"不支持的特征类型: {facet}")

    # 遍历模型的每个块
    for block_idx, block in enumerate(model.blocks):
        if block_idx in layers:
            # 根据不同的特征类型注册相应的钩子
            if facet == 'token':
                # 注册token级特征的钩子
                hook_handlers.append(
                    block.register_forward_hook(get_hook(model, facet, feats))
                )
            elif facet == 'attn':
                # 注册注意力特征的钩子，需要考虑xFormers的可用性
                if XFORMERS_AVAILABLE:
                    hook_handlers.append(
                        block.attn.register_forward_hook(get_hook(model, facet, feats))
                    )
                else:
                    hook_handlers.append(
                        block.attn.attn_drop.register_forward_hook(get_hook(model, facet, feats))
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
    model, 
    batch: torch.Tensor, 
    layers: List[int], 
    facet: str = 'key'
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
        register_hooks(model, layers, facet, hook_handlers, feats)
        # 前向传播
        _ = model(batch)
    finally:
        # 确保钩子被移除，即使发生异常
        unregister_hooks(hook_handlers)
    
    return feats

def extract_saliency_maps(model, layers: List[int], batch: torch.Tensor) -> torch.Tensor:
    """
    提取显著性图（Saliency Maps）
    
    通过提取模型最后一层CLS token的注意力头的平均值来生成显著性图。
    所有值都被归一化到0到1之间。
    
    Args:
        model: 要提取特征的模型实例
        layers: 要提取特征的层索引列表
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
    feats = extract_features(model, batch, layers, 'attn')
    
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
    cls_attn_maps = (cls_attn_map - temp_mins.unsqueeze(1)) / (temp_maxs - temp_mins).unsqueeze(1)
    
    return cls_attn_maps

