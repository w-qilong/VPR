import torch
# from .nearest_neighbor_selector import (
#     compute_mutual_nearest_neighbors_batch_parallel,
#     extract_feature_windows,
#     compute_window_similarities,
# )

import torch.nn.functional as F
import numpy as np

def match_batch_tensor(fm1, fm2, trainflag, grid_size):
    '''特征匹配函数
    参数:
    fm1: (l,D) - 查询图像特征
    fm2: (N,l,D) - N张数据库图像特征
    '''
    # 计算特征相似度矩阵
    M = torch.matmul(fm2, fm1.T) # (N,l,l) 

    # 互最近邻匹配
    max1 = torch.argmax(M, dim=1) # (N,l) fm1中每个点在fm2中的最佳匹配
    max2 = torch.argmax(M, dim=2) # (N,l) fm2中每个点在fm1中的最佳匹配
    m = max2[torch.arange(M.shape[0]).reshape((-1,1)), max1] # (N,l) 验证互最近邻
    valid = torch.arange(M.shape[-1]).repeat((M.shape[0],1)).cuda() == m # (N,l) 布尔掩码

    scores = torch.zeros(fm2.shape[0]).cuda()

    # 对每张图像计算匹配分数
    for i in range(fm2.shape[0]):
        idx1 = torch.nonzero(valid[i,:]).squeeze()  # 获取有效匹配点在fm1中的索引
        idx2 = max1[i,:][idx1]  # 获取对应在fm2中的索引
        assert idx1.shape==idx2.shape

        if trainflag:  # 训练模式
            if len(idx1.shape)>0:      
                # 计算匹配点对的特征相似度平均值
                similarity = torch.mean(torch.sum(fm1[idx1] * fm2[i][idx2],dim=1),dim=0)
            else:
                print("No mutual nearest neighbors!")
                similarity = torch.mean(torch.sum(fm1 * fm2[i],dim=1),dim=0)
            return similarity
        else:  # 测试模式
            # 使用匹配点对数量作为相似度分数
            scores[i] = 0 if len(idx1.shape)<1 else len(idx1)
    return scores

def local_sim(features_1, features_2, trainflag=False):
    '''局部特征相似度计算主函数
    参数:
    features_1: 查询图像特征 
    features_2: 数据库图像特征
    trainflag: 是否为训练模式
    '''
    B, H, W, C = features_2.shape  # B:批次大小, H,W:特征图尺寸, C:特征维度
    
    if trainflag:  # 训练模式
        queries = features_1
        preds = features_2
        # 重塑特征维度为(B, H*W, C)
        queries,preds = queries.view(B, H*W, C),preds.view(B, H*W, C)
        similarity = torch.zeros(B).cuda()
        # 逐图像对计算相似度
        for i in range(B):
            query,pred = queries[i],preds[i].unsqueeze(0)
            similarity[i] = match_batch_tensor(query, pred, trainflag, grid_size=(H, W))
        return similarity
    else:  # 测试模式
        query = features_1
        preds = features_2
        # 重塑特征维度
        query,preds = query.view(H*W, C),preds.view(B, H*W, C)
        scores = match_batch_tensor(query, preds,trainflag, grid_size=(H, W))
        return scores
    

class LocalFeatureLoss(torch.nn.Module):
    def __init__(self):
        super(LocalFeatureLoss,self).__init__()
        return
    def forward(self, feats1, feats2, trainflag=True):
        similarity = local_sim(feats1,feats2,trainflag=trainflag)
        return similarity

# class PairwiseLocalMatchingLoss(torch.nn.Module):
#     """局部匹配损失函数
    
#     该类实现了一个基于局部特征匹配的损失函数，用于比较两张图像中对应区域的相似度。
    
#     参数:
#         topk (int): 选择互相最近邻时要考虑的最近邻数量
#     """
#     def __init__(self, topk=8):
#         super(PairwiseLocalMatchingLoss, self).__init__()
#         self.topk = topk

#     def forward(
#         self,
#         topk_patch_tokens1,  # 第一张图像的patch token特征 [B, N, D]
#         topk_patch_tokens2,  # 第二张图像的patch token特征 [B, N, D]
#         conv_features1,      # 第一张图像的卷积特征图 [B, C, H, W]
#         conv_features2,      # 第二张图像的卷积特征图 [B, C, H, W]
#         window_size=8,       # 特征窗口大小
#         is_training=True,    # 是否处于训练模式
#     ):
#         """前向传播函数
        
#         步骤:
#         1. 计算两组特征之间的互相最近邻
#         2. 根据最近邻位置提取特征窗口
#         3. 计算对应窗口之间的相似度
        
#         返回:
#             torch.Tensor: 计算得到的损失值
#         """
#         # 计算特征之间的互相最近邻
#         mutual_neighbors = compute_mutual_nearest_neighbors_batch_parallel(
#             topk_patch_tokens1, topk_patch_tokens2, topk=self.topk
#         )
#         # 提取特征窗口
#         windows1, windows2 = extract_feature_windows(
#             conv_features1, conv_features2, mutual_neighbors, window_size=window_size
#         )
#         # 计算窗口相似度并返回损失值
#         return compute_window_similarities(windows1, windows2, is_training)

if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    torch.manual_seed(42)
    
    # 测试参数
    batch_size = 4
    feature_size = 112  # 特征图的高度和宽度
    channels = 384     # 特征通道数
    
    # 创建测试数据
    anchor = torch.randn(batch_size, feature_size, feature_size, channels).cuda()
    positive = torch.randn(batch_size, feature_size, feature_size, channels).cuda()
    negative = torch.randn(batch_size, feature_size, feature_size, channels).cuda()
    
    # 创建损失函数实例
    loss_fn = LocalFeatureLoss()
    
    # 计算损失
    feature_data = [anchor, positive, negative]
    loss = loss_fn(anchor,positive)
    
    print("=== LocalFeatureLoss测试 ===")
    print(f"输入特征形状: {anchor.shape}")
    print(f"损失值: {loss}")
    
    # # 测试local_sim函数
    # print("\n=== local_sim函数测试 ===")
    # sim_pos = local_sim(anchor, positive, trainflag=True)
    # sim_neg = local_sim(anchor, negative, trainflag=True)
    # print(f"正样本相似度: {sim_pos}")
    # print(f"负样本相似度: {sim_neg}")
