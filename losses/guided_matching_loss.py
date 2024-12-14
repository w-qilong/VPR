import torch
import torch.nn as nn


def compute_guided_matching(topk_features1, topk_features2, is_training=True):
    """
    计算两组特征之间的引导匹配和相似度
    
    Args:
        topk_features1: 第一张图片的top-k特征 [B, C, top_k]
        topk_features2: 第二张图片的top-k特征 [B, C, top_k]
        top_k: 特征点数量，默认64
        is_training: 是否为训练模式，默认True
    
    Returns:
        training模式: 批次平均相似度 (scalar)
        testing模式: 每个样本的有效匹配数 [B]
    """
    B, C, top_k = topk_features1.shape
    device = topk_features1.device
    
    # 计算特征相似度矩阵 [B, top_k, top_k]
    similarity_matrix = torch.bmm(topk_features1.transpose(1, 2), topk_features2)
    
    # 计算互最近邻匹配
    _, matches1 = similarity_matrix.max(dim=2)  # [B, top_k]
    _, matches2 = similarity_matrix.max(dim=1)  # [B, top_k]
    
    # 验证互最近邻关系
    mutual_matches = matches2.gather(1, matches1) == torch.arange(
        top_k, device=device).unsqueeze(0)
    
    if not is_training:
        return mutual_matches.sum(1).float()
    
    # 训练模式：计算匹配点对的平均相似度
    has_mutual_matches = mutual_matches.any(dim=1)
    batch_similarities = torch.zeros(B, device=device)
    
    if has_mutual_matches.any():
        # 获取匹配点对的相似度
        valid_similarities = torch.gather(
            similarity_matrix, 
            2, 
            matches1.unsqueeze(2)
        ).squeeze(2) * mutual_matches.float()
        
        # 计算有效匹配的平均相似度
        match_counts = mutual_matches[has_mutual_matches].sum(1).clamp(min=1e-6)
        batch_similarities[has_mutual_matches] = (
            valid_similarities[has_mutual_matches].sum(1) / match_counts
        )
    
    # 处理没有互最近邻匹配的样本
    if (~has_mutual_matches).any():
        batch_similarities[~has_mutual_matches] = (
            similarity_matrix[~has_mutual_matches].mean(dim=(1,2))
        )
    
    return batch_similarities.mean()

def test_guided_matching():
    # 设置随机种子保证结果可复现
    torch.manual_seed(42)
    
    # 测试参数
    batch_size = 2
    channels = 256
    top_k = 64
    
    # 创建随机测试数据
    features1 = torch.randn(batch_size, channels, top_k)
    features2 = torch.randn(batch_size, channels, top_k)
    
    # 归一化特征向量（实际应用中通常需要归一化）
    features1 = nn.functional.normalize(features1, dim=1)
    features2 = nn.functional.normalize(features2, dim=1)
    
    print("测试训练模式:")
    train_result = compute_guided_matching(features1, features2, is_training=True)
    print(f"训练模式下的平均相似度: {train_result:.4f}")
    
    print("\n测试测试模式:")
    test_result = compute_guided_matching(features1, features2, is_training=False)
    print(f"每个样本的有效匹配数: {test_result}")
    
    # 测试边界情况：完全相同的特征
    print("\n测试完全匹配的情况:")
    same_features = torch.randn(batch_size, channels, top_k)
    same_features = nn.functional.normalize(same_features, dim=1)
    train_result_same = compute_guided_matching(same_features, same_features, is_training=True)
    test_result_same = compute_guided_matching(same_features, same_features, is_training=False)
    print(f"完全匹配时的平均相似度: {train_result_same:.4f}")
    print(f"完全匹配时的有效匹配数: {test_result_same}")

if __name__ == "__main__":
    test_guided_matching()