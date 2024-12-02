import torch

# 批量处理版本（使用并行计算）
def compute_mutual_nearest_neighbors_batch_parallel(topk_patch_tokens1, topk_patch_tokens2, topk=8):
    """
    基于DINO输出计算两组特征张量之间的互相最近邻。

    :param features1: 第一组特征张量，形状为 (batch_size, num_patches, embedding_dim)
    :param features2: 第二组特征张量，形状为 (batch_size, num_patches, embedding_dim)
    :param important_indices1: 第一组图像中重要patch的索引，形状为 (batch_size, K)
    :param important_indices2: 第二组图像中重要patch的索引，形状为 (batch_size, K)
    :param topk: 返回的最近邻数量
    :return: 每个批次的互相最近邻列表 [[(idx1, idx2, sim), ...], ...]
    """
    assert topk_patch_tokens1.shape == topk_patch_tokens2.shape, "特征维度不匹配"
    assert topk_patch_tokens1.dim() == 3, "特征张量维度应为3"

    # 特征归一化 (batch_size, num_patches, embedding_dim)
    topk_patch_tokens1_norm = torch.nn.functional.normalize(topk_patch_tokens1, p=2, dim=2)
    topk_patch_tokens2_norm = torch.nn.functional.normalize(topk_patch_tokens2, p=2, dim=2)
    
    # 批量计算余弦相似度
    # 使用einsum进行批量矩阵乘法
    similarity = torch.einsum('bik,bjk->bij', topk_patch_tokens1_norm, topk_patch_tokens2_norm)
    
    # 获取从topk_patch_tokens1到topk_patch_tokens2的topk最近邻
    distances1, indices1 = torch.topk(similarity, 1, dim=2)
    
    # 获取从topk_patch_tokens2到topk_patch_tokens1的topk最近邻
    distances2, indices2 = torch.topk(similarity.transpose(1, 2), 1, dim=2)
    
    batch_size = topk_patch_tokens1.shape[0]
    all_mutual_neighbors = []
    
     # 简化的互相最近邻检查
    for b in range(batch_size):
        batch_mutual_neighbors = []
        for i in range(topk_patch_tokens1.shape[1]):
            j = indices1[b, i, 0]  # 只取第一个最近邻
            # 检查是否互为最近邻
            if i == indices2[b, j, 0]:
                batch_mutual_neighbors.append(
                    (i, j.item(), distances1[b, i, 0].item())
                )
        
        # 按相似度降序排序
        batch_mutual_neighbors.sort(key=lambda x: x[2], reverse=True)

        # 只保留前topk个最相似的配对
        if len(batch_mutual_neighbors) > topk:
            batch_mutual_neighbors = batch_mutual_neighbors[:topk]
        all_mutual_neighbors.append(batch_mutual_neighbors)
    
    return all_mutual_neighbors


def extract_feature_windows(conv_features1, conv_features2, mutual_neighbors, window_size=8):
    """
    从两组卷积特征图中根据最近邻索引提取对应的窗口

    Args:
        conv_features1: 第一组卷积特征图 (batch_size, channels, 112, 112)
        conv_features2: 第二组卷积特征图 (batch_size, channels, 112, 112)
        mutual_neighbors: compute_mutual_nearest_neighbors_batch_parallel返回的最近邻列表
                        格式为 [[(idx1, idx2, sim), ...], ...]
        window_size: 要提取的窗口大小
    
    Returns:
        windows1, windows2: 两组提取的特征窗口
        每组形状为 (batch_size, num_pairs, channels, window_size, window_size)

    计算过程说明:
    1. 输入参数示例:
       - 输入图像尺寸: 224×224
       - 特征图尺寸: 112×112 (经过卷积层下采样后)
       - patch数量: 16×16 (每行16个patch，每列16个patch)
       - patch大小: 7×7 (在特征图上)
       - 窗口大小: 16×16 (要提取的局部特征大小)

    2. 对于每个patch的处理步骤:
       a) 计算patch中心点坐标:
          x = (idx % num_patches) * patch_size + patch_size//2
          y = (idx // num_patches) * patch_size + patch_size//2
          
          例如，对于第3行第4列的patch(索引52):
          x = (52 % 16) * 7 + 7//2 = 4 * 7 + 3 = 31
          y = (52 // 16) * 7 + 7//2 = 3 * 7 + 3 = 24

       b) padding处理:
          pad_size = window_size // 2 = 8
          x_pad = x + pad_size = 31 + 8 = 39
          y_pad = y + pad_size = 24 + 8 = 32

       c) 窗口范围计算:
          x_start = x_pad - window_size//2 = 39 - 8 = 31
          x_end = x_pad + window_size//2 = 39 + 8 = 47
          y_start = y_pad - window_size//2 = 32 - 8 = 24
          y_end = y_pad + window_size//2 = 32 + 8 = 40

    3. 输出张量形状示例:
       假设 batch_size=4, topk=5:
       输出shape为(4, 5, 128, 16, 16)
       - 4: batch_size
       - 5: 每个样本选择的topk个最近邻对数
       - 128: 特征通道数
       - 16x16: 提取的窗口大小

    4. 提取窗口的代码实现:
        这个过程确保了：
        对每个最近邻patch对，我们都提取了以其中心为中心的局部特征窗口
        通过padding处理确保了边界patch也能提取完整的窗口
        保持了batch处理能力，提高计算效率
    """
    batch_size = conv_features1.shape[0]
    channels = conv_features1.shape[1]  # 128
    feature_size = conv_features1.shape[2]  # 112
    
    # 对于224x224的输入和14x14的patch:
    patch_size = 7  # 特征图上每个patch对应的大小
    num_patches = 16  # 每行/列的patch数量
    
    # 计算padding大小
    pad_size = window_size // 2
    padded_features1 = torch.nn.functional.pad(
        conv_features1,
        (pad_size, pad_size, pad_size, pad_size),
        mode='reflect'
    )
    padded_features2 = torch.nn.functional.pad(
        conv_features2,
        (pad_size, pad_size, pad_size, pad_size),
        mode='reflect'
    )
    
    all_windows1 = []
    all_windows2 = []
    
    for b in range(batch_size):
        # 检查当前batch的mutual_neighbors是否为空
        if not mutual_neighbors[b]:
            # 如果为空，创建形状为(0, channels, window_size, window_size)的空张量
            print(f"Batch {b} 没有找到互相最近邻")
            empty_windows = torch.zeros(0, channels, window_size, window_size, 
                                     device=conv_features1.device)
            all_windows1.append(empty_windows)
            all_windows2.append(empty_windows)
            continue

        batch_windows1 = []
        batch_windows2 = []
        
        for idx1, idx2, _ in mutual_neighbors[b]:
            # 处理第一组特征的窗口
            x1 = (idx1 % num_patches) * patch_size + patch_size // 2
            y1 = (idx1 // num_patches) * patch_size + patch_size // 2
            x1_pad = x1 + pad_size
            y1_pad = y1 + pad_size
            
            # 处理第二组特征的窗口
            x2 = (idx2 % num_patches) * patch_size + patch_size // 2
            y2 = (idx2 // num_patches) * patch_size + patch_size // 2
            x2_pad = x2 + pad_size
            y2_pad = y2 + pad_size
            
            # 提取两组窗口
            window1 = padded_features1[b:b+1, :,
                     y1_pad-window_size//2:y1_pad+window_size//2,
                     x1_pad-window_size//2:x1_pad+window_size//2]
            
            window2 = padded_features2[b:b+1, :,
                     y2_pad-window_size//2:y2_pad+window_size//2,
                     x2_pad-window_size//2:x2_pad+window_size//2]
            
            batch_windows1.append(window1)
            batch_windows2.append(window2)
        
        # 将batch中的所有窗口堆叠
        batch_windows1 = torch.cat(batch_windows1, dim=0)
        batch_windows2 = torch.cat(batch_windows2, dim=0)
        
        all_windows1.append(batch_windows1)
        all_windows2.append(batch_windows2)
    
    # 将所有batch的窗口堆叠
    # 注意：如果某个batch的windows为空，stack操作仍然有效
    return torch.stack(all_windows1), torch.stack(all_windows2)


def compute_window_similarities(windows1, windows2, is_training=True):
    """
    计算两组特征窗口之间的相似性

    Args:
        windows1: 第一组特征窗口 (batch_size, num_pairs, channels, window_size, window_size)
        windows2: 第二组特征窗口 (batch_size, num_pairs, channels, window_size, window_size)
        is_training: 是否为训练模式

    Returns:
        training模式: 所有窗口对的平均相似度 (scalar)
        inference模式: 所有匹配的窗口中最近邻像素对的数量 (scalar)
    """
    # 获取batch大小
    batch_size = windows1.shape[0]

    # 初始化batch平均相似度
    total_similarity = 0

    # 初始化匹配数，用于存储batch中每个pair的匹配数
    all_batch_matches = torch.zeros(batch_size, device=windows1.device)
    
    for b in range(batch_size):
        # 跳过空的batch
        if windows1[b].shape[0] == 0:
            continue
            
        # 将特征展平为 (num_pairs, channels, window_pixels)
        feat1 = windows1[b].view(windows1[b].shape[0], windows1[b].shape[1], -1)
        feat2 = windows2[b].view(windows2[b].shape[0], windows2[b].shape[1], -1)
        
        # 计算特征相似度矩阵 (num_pairs, window_pixels, window_pixels)
        M = torch.matmul(feat1.transpose(1, 2), feat2)

        # 获取特征相似度矩阵中每个像素位置的最佳匹配
        max1 = torch.argmax(M, dim=1)  # (num_pairs, window_pixels)
        max2 = torch.argmax(M, dim=2)  # (num_pairs, window_pixels)

        # 检查互相最近邻
        # 创建与max1相同形状的索引矩阵
        idx = torch.arange(M.shape[2], device=M.device)
        idx = idx.view(1, -1).expand(M.shape[0], -1)  # (num_pairs, window_pixels)
        
        # 找到互相最近邻的位置
        mutual_nearest = (idx == max2.gather(1, max1))  # (num_pairs, window_pixels)

        # 初始化当前batch的匹配结果
        pair_matches = torch.zeros(M.shape[0], device=windows1.device)

        # 为batch中每个pair计算相似度或匹配数
        for i in range(M.shape[0]):
            # 如果训练模式，计算互相最近邻的相似度
            if is_training:
                if mutual_nearest[i].any():  # 确保有互最近邻点
                    # 计算互相最近邻位置的相似度
                    similarities = torch.gather(M[i], 1, max1[i].unsqueeze(1)).squeeze(1)  # (window_pixels,)
                    valid_similarities = similarities[mutual_nearest[i]]
                    if valid_similarities.numel() > 0:
                        total_similarity += valid_similarities.mean().item()
                    else:
                        # 如果没有互最近邻，使用全局相似度
                        similarity = torch.mean(M[i])
                        total_similarity += similarity.item()
                else:
                    print("No mutual nearest neighbors!")
                    similarity = torch.mean(M[i])
                    total_similarity += similarity.item()
            else:
                # 统计当前pair的互相最近邻数量
                pair_matches[i] = mutual_nearest[i].sum().item()

        # 计算当前batch的匹配数
        num_matches = pair_matches.sum().item()
        all_batch_matches[b] = num_matches

    if is_training:
        # 计算平均相似度时考虑实际的pair数量
        total_pairs = sum(len(windows1[b]) for b in range(batch_size))
        return total_similarity / max(total_pairs, 1)  # 避免除零错误
    else:
        return all_batch_matches    


# # 使用示例
# if __name__ == "__main__":
#     # 设置随机种子以确保结果可重现
#     torch.manual_seed(42)
    
#     # 测试参数
#     batch_size = 2
#     num_patches = 256
#     embedding_dim = 128
#     topk = 3
#     window_size = 8
#     num_important = 50  # 重要patch的数量
    
#     # 创建测试数据
#     features1 = torch.randn(batch_size, num_patches, embedding_dim)
#     features2 = torch.randn(batch_size, num_patches, embedding_dim)
#     conv_features1 = torch.randn(batch_size, 128, 112, 112)
#     conv_features2 = torch.randn(batch_size, 128, 112, 112)
    
#     # 创建重要patch的索引
#     important_indices1 = torch.randint(0, num_patches, (batch_size, num_important))
#     important_indices2 = torch.randint(0, num_patches, (batch_size, num_important))
    
#     print("=== 测试互相最近邻计算 ===")
#     print("\n1. 使用所有patches:")
#     mutual_pairs_all = compute_mutual_nearest_neighbors_batch_parallel(
#         features1, features2, topk=topk
#     )
    
#     print("\n2. 只使用重要patches:")
#     mutual_pairs_important = compute_mutual_nearest_neighbors_batch_parallel(
#         features1, features2, topk=topk
#     )
    
#     # 打印结果
#     for test_name, mutual_pairs in [("所有patches", mutual_pairs_all), 
#                                   ("重要patches", mutual_pairs_important)]:
#         print(f"\n测试 {test_name}:")
#         for b, pairs in enumerate(mutual_pairs):
#             print(f"\n批次 {b}:")
#             print(f"找到 {len(pairs)} 对互相最近邻")
#             for i, (idx1, idx2, sim) in enumerate(pairs):
#                 print(f"  对 {i+1}: patch1={idx1}, patch2={idx2}, 相似度={sim:.4f}")

#     print("\n=== 测试特征窗口提取 ===")
#     windows1, windows2 = extract_feature_windows(
#         conv_features1, conv_features2, 
#         mutual_pairs_important,  # 使用重要patches的结果
#         window_size
#     )
#     print(f"窗口1形状: {windows1.shape}")
#     print(f"窗口2形状: {windows2.shape}")

#     print("\n=== 测试相似度计算 ===")
#     # 训练模式测试
#     train_sim = compute_window_similarities(windows1, windows2, is_training=True)
#     print(f"训练模式 - 平均相似度: {train_sim:.4f}")
    
#     # 推理模式测试
#     infer_matches = compute_window_similarities(windows1, windows2, is_training=False)
#     print(f"推理模式 - 每个batch的匹配数:")
#     for b in range(batch_size):
#         print(f"  批次 {b}: {infer_matches[b]:.0f} 个匹配")

