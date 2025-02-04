from typing import List
import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable
import torch
import time
# from losses import compute_guided_matching
from .hook_func import find_correspondences,chunk_cosine_sim,extract_descriptors,log_bin,extract_saliency_maps,extract_features


def get_validation_recalls(
    r_list,
    q_list,
    k_values,
    gt,
    print_results=True,
    faiss_gpu=False,
    dataset_name="dataset without name ?",
):
    embed_size = r_list.shape[1]
    if faiss_gpu:
        res = faiss.StandardGpuResources()
        flat_config = faiss.GpuIndexFlatConfig()
        flat_config.useFloat16 = True
        flat_config.device = 0
        faiss_index = faiss.GpuIndexFlatL2(res, embed_size, flat_config)
    # build index
    else:
        faiss_index = faiss.IndexFlatL2(embed_size)

    # add references
    faiss_index.add(r_list.float())

    # search for queries in the index
    # predictions: 包含每个q_list对应的r_list中top k个最相似结果的index
    _, predictions = faiss_index.search(
        q_list.float(), max(k_values)
    )  # predictions为q_list中每个查询对应的topk参考结果索引

    # start calculating recall_at_k
    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], gt[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = correct_at_k / len(predictions)
    d = {k: np.round(v*100, 2) for (k, v) in zip(k_values, correct_at_k)}

    if print_results:
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ["K"] + [str(k) for k in k_values]
        table.add_row(["Recall@K"] + [f"{100 * v:.2f}" for v in correct_at_k])
        print(table.get_string(title=f"Performances on {dataset_name}"))

    return d, predictions


# 定义函数获取两张图像的最近邻对应点对数量
def get_correspondences(
    query_desc: torch.Tensor,      # shape: [B, 1, N, D]
    ref_desc: torch.Tensor,        # shape: [B, 1, N, D]
    query_attn_maps: torch.Tensor, # shape: [B, N]
    ref_attn_maps: torch.Tensor,   # shape: [B, N]
    saliency_thresh: float = 0.05,
    nn_match_thresh: float = 0.65,
) -> torch.Tensor:
    """计算两张图像之间的有效对应点对数量。
    
    通过以下步骤筛选有效对应点:
    1. 使用显著性阈值过滤，只保留显著性高的区域
    2. 计算特征描述符之间的余弦相似度
    3. 找出互为最近邻的匹配点对(Best Buddies)
    4. 应用相似度阈值过滤
    
    Args:
        query_desc: 查询图像特征描述符
        ref_desc: 参考图像特征描述符 
        query_attn_maps: 查询图像显著性图
        ref_attn_maps: 参考图像显著性图
        saliency_thresh: 显著性阈值，用于过滤低显著性区域
        nn_match_thresh: 最近邻匹配阈值，用于过滤低相似度的匹配
    
    Returns:
        torch.Tensor: 每个批次中有效对应点对的数量, shape: [B]
    """
    device = query_desc.device
    # 获取实际的patch数量
    total_patches = query_desc.shape[2]

    # 2. 提取并处理显著性图，生成前景掩码
    query_fg_mask = query_attn_maps > saliency_thresh  # shape: [N], bool tensor
    ref_fg_mask = ref_attn_maps > saliency_thresh     # shape: [N], bool tensor

    # 3. 计算特征描述符间的余弦相似度
    query_desc=query_desc.float()
    ref_desc=ref_desc.float()
    similarities = chunk_cosine_sim(query_desc, ref_desc)  # shape: [B, 1, N, N]

    # 找到相互最近邻的点对
    sim_q2r, nn_q2r = torch.max(similarities, dim=-1)  # shapes: [B, 1, N], [B, 1, N]
    sim_r2q, nn_r2q = torch.max(similarities, dim=-2)  # shapes: [B, 1, N], [B, 1, N]

    # 移除多余的维度1，但保留批次维度
    sim_q2r = sim_q2r.squeeze(1)    # shape: [B, N]
    nn_q2r = nn_q2r.squeeze(1)      # shape: [B, N]
    sim_r2q = sim_r2q.squeeze(1)    # shape: [B, N]
    nn_r2q = nn_r2q.squeeze(1)      # shape: [B, N]

    # 添加相似度阈值过滤
    sim_mask = (sim_q2r > nn_match_thresh)  # 只保留相似度大于阈值的点对

    # 寻找互为最近邻的匹配点对（Best Buddies）
    # 扩展patch_indices到批次维度
    patch_indices = torch.arange(total_patches, device=device)[None].expand(query_desc.shape[0], -1)  # shape: [B, N]

    best_buddies_mask = torch.gather(nn_r2q, 1, nn_q2r) == patch_indices  # shape: [B, N]
    best_buddies_mask = torch.bitwise_and(best_buddies_mask, sim_mask)  # shape: [B, N]

    # 扩展前景掩码到批次维度
    ref_fg_mapped = torch.gather(nn_r2q, 1, torch.where(ref_fg_mask)[1][None].expand(query_desc.shape[0], -1))
    ref_fg_mask_mapped = torch.zeros((query_desc.shape[0], total_patches), dtype=torch.bool, device=device)
    ref_fg_mask_mapped.scatter_(1, ref_fg_mapped, True)
    
    # 应用前景掩码过滤
    valid_matches = torch.bitwise_and(best_buddies_mask, query_fg_mask)
    valid_matches = torch.bitwise_and(valid_matches, ref_fg_mask_mapped)

    return valid_matches.sum(dim=1)  # 返回每个批次的有效匹配数，shape: [B]

def single_rerank(
    query_desc,
    ref_desc,
    query_attn_maps,
    ref_attn_maps,
    first_predictions,
    saliency_thresh=0.05,
    nn_match_thresh=0.75
):
    """
    批量处理版本的重排序函数
    
    Args:
        query_desc: [1, 1, P, D] 查询图像描述符
        ref_desc: [M, 1, P, D] 参考图像描述符
        query_attn_maps: [1, P] 查询图像显著性图
        ref_attn_maps: [M, P] 参考图像显著性图
        first_predictions: [M] 第一阶段检索的topK index结果
        saliency_thresh: 显著性阈值，用于过滤低显著性区域
        nn_match_thresh: 最近邻匹配阈值，用于过滤低相似度的匹配
    """
    # 遍历每个查询图像
    nn_pairs=get_correspondences(query_desc, ref_desc, query_attn_maps,
                                  ref_attn_maps, saliency_thresh=saliency_thresh, nn_match_thresh=nn_match_thresh)
    nn_pairs=nn_pairs.cpu().numpy()
    rerank_index= nn_pairs.argsort()[::-1].copy()
    rerank_predictions=first_predictions[rerank_index]
    return rerank_predictions


def get_rerank_results(
    ref_topk_local_features,
    query_topk_local_features,
    first_predictions,
    ground_truth,
    k_values,
    print_results=True,
    dataset_name="dataset without name ?",
):
    # 初始化重排序结果
    rerank_predictions = []

    # 遍历每个查询图像
    for index, pre in enumerate(first_predictions):
        # 获取对应的参考图像的topk局部特征
        cur_ref_topk_local_features = ref_topk_local_features[pre]
        # 获取对应的查询图像的topk局部特征
        cur_query_topk_local_features = query_topk_local_features[index]
        # 重排序
        cur_ref_topk_local_features = torch.tensor(cur_ref_topk_local_features).cuda() # 50 128 64
        cur_query_topk_local_features = torch.tensor(cur_query_topk_local_features).unsqueeze(0).repeat(cur_ref_topk_local_features.shape[0],1,1).cuda() # 128 64
        # 计算引导匹配
        with torch.no_grad():
            num_local_matches = compute_guided_matching(
                cur_query_topk_local_features,
                cur_ref_topk_local_features, 
                is_training=False,
            )
        rerank_index=num_local_matches.cpu().numpy().argsort()[::-1].copy()
        rerank_predictions.append(pre[rerank_index])
    
    # 计算recall@k
    predictions=np.array(rerank_predictions)
     # start calculating recall_at_k
    correct_at_k = np.zeros(len(k_values))
    for q_idx, pred in enumerate(predictions):
        for i, n in enumerate(k_values):
            # if in top N then also in top NN, where NN > N
            if np.any(np.in1d(pred[:n], ground_truth[q_idx])):
                correct_at_k[i:] += 1
                break

    correct_at_k = correct_at_k / len(predictions)
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}


    if print_results:
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ["K"] + [str(k) for k in k_values]
        table.add_row(["Recall@K"] + [f"{100 * v:.2f}" for v in correct_at_k])
        print(table.get_string(title=f"Performances rerank on {dataset_name}"))

    return d, rerank_predictions

if __name__ == "__main__":
    # 创建一组确定的测试数据
    B, N, D = 1, 4, 8  # 使用小规模数据便于验证
    
    # 创建特征描述符
    # 在每个批次中设置一些确定的匹配点对
    query_desc = torch.zeros(B, 1, N, D).cuda()
    ref_desc = torch.zeros(B, 1, N, D).cuda()
    
    # 批次1的匹配点对设置
    # 点0和点0匹配，点1和点1匹配
    query_desc[0, 0, 0] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
    query_desc[0, 0, 1] = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0])
    ref_desc[0, 0, 0] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
    ref_desc[0, 0, 1] = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0])
    
    # 批次2的匹配点对设置
    # 点0和点0匹配，点1和点2匹配，点2和点1匹配
    query_desc[1, 0, 0] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
    query_desc[1, 0, 1] = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0])
    query_desc[1, 0, 2] = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0])
    ref_desc[1, 0, 0] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
    ref_desc[1, 0, 2] = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0])
    ref_desc[1, 0, 1] = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0])
    
    # 创建显著性图
    # 将匹配点对的显著性值设置为高于阈值
    query_attn_maps = torch.ones(B, N).cuda() * 0.03  # 默认值低于阈值
    ref_attn_maps = torch.ones(B, N).cuda() * 0.03
    
    # 设置匹配点对的显著性值
    query_attn_maps[0, :2] = 0.1  # 批次1的前两个点
    ref_attn_maps[0, :2] = 0.1
    
    query_attn_maps[1, :3] = 0.1  # 批次2的前三个点
    ref_attn_maps[1, :3] = 0.1
    
    # 测试函数
    matches = get_correspondences(
        query_desc=query_desc,
        ref_desc=ref_desc,
        query_attn_maps=query_attn_maps,
        ref_attn_maps=ref_attn_maps,
        saliency_thresh=0.05,
        nn_match_thresh=0.65
    )
    
    print("测试结果:")
    print(f"输入形状:")
    print(f"- query_desc: {query_desc.shape}")
    print(f"- ref_desc: {ref_desc.shape}")
    print(f"- query_attn_maps: {query_attn_maps.shape}")
    print(f"- ref_attn_maps: {ref_attn_maps.shape}")
    print(f"\n每个批次的匹配点对数量: {matches}")
    print(f"输出形状: {matches.shape}")
    
    # 预期结果说明
    print("\n预期结果说明:")
    print("批次1应该有2个匹配点对 (0-0, 1-1)")
    print("批次2应该有3个匹配点对 (0-0, 1-2, 2-1)")

    # 测试 single_rerank 函数
    print("\n开始测试 single_rerank 函数:")
    
    # 准备测试数据
    M = 3  # 参考图像数量
    P = 4  # 每张图像的特征点数量
    D = 8  # 特征维度
    
    # 创建查询图像特征
    query_desc_rerank = torch.zeros(1, 1, P, D).cuda()
    query_desc_rerank[0, 0, 0] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
    query_desc_rerank[0, 0, 1] = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0])
    
    # 创建参考图像特征
    ref_desc_rerank = torch.zeros(M, 1, P, D).cuda()
    # 第一张参考图像（最佳匹配）
    ref_desc_rerank[0, 0, 0] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
    ref_desc_rerank[0, 0, 1] = torch.tensor([0, 1, 0, 0, 0, 0, 0, 0])
    # 第二张参考图像（次佳匹配）
    ref_desc_rerank[1, 0, 0] = torch.tensor([1, 0, 0, 0, 0, 0, 0, 0])
    # 第三张参考图像（最差匹配）
    ref_desc_rerank[2, 0, 1] = torch.tensor([0, 0, 1, 0, 0, 0, 0, 0])

    ref_desc_rerank = ref_desc_rerank.flip(0)
    
    # 创建显著性图
    query_attn_maps_rerank = torch.ones(1, P).cuda() * 0.1  # 所有点都高于阈值
    ref_attn_maps_rerank = torch.ones(M, P).cuda() * 0.1
    
    # 创建第一阶段的预测结果
    first_predictions = np.array([2, 1, 0])  # 原始排序：[2, 1, 0]
    
    # 执行重排序
    reranked_predictions = single_rerank(
        query_desc_rerank,
        ref_desc_rerank,
        query_attn_maps_rerank,
        ref_attn_maps_rerank,
        first_predictions,
        saliency_thresh=0.05,
        nn_match_thresh=0.65
    )
    
    print(f"原始排序: {first_predictions}")
    print(f"重排序结果: {reranked_predictions}")
    print("预期结果说明:")
    print("- 图像0应该排在最前（有2个匹配点）")
    print("- 图像1应该排在第二（有1个匹配点）")
    print("- 图像2应该排在最后（没有匹配点）")
