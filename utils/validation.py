import numpy as np
import faiss
import faiss.contrib.torch_utils
from prettytable import PrettyTable
import torch

from losses import compute_guided_matching


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
    faiss_index.add(r_list)

    # search for queries in the index
    # predictions: 包含每个q_list对应的r_list中top k个最相似结果的index
    _, predictions = faiss_index.search(
        q_list, max(k_values)
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
    d = {k: v for (k, v) in zip(k_values, correct_at_k)}

    if print_results:
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ["K"] + [str(k) for k in k_values]
        table.add_row(["Recall@K"] + [f"{100 * v:.2f}" for v in correct_at_k])
        print(table.get_string(title=f"Performances on {dataset_name}"))

    return d, predictions


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
    rerank_results = []

    # 遍历每个查询图像
    for index, pre in enumerate(first_predictions):
        # 获取对应的参考图像的topk局部特征
        ref_topk_local_features = ref_topk_local_features[pre]
        # 获取对应的查询图像的topk局部特征
        query_topk_local_features = query_topk_local_features[index]
        # 重排序
        ref_topk_local_features = torch.tensor(ref_topk_local_features).cuda()
        query_topk_local_features = torch.tensor(query_topk_local_features).cuda()
        # 计算引导匹配
        with torch.no_grad():
            num_local_matches = compute_guided_matching(
                query_topk_local_features.unsqueeze(0),
                ref_topk_local_features,
                is_training=False,
            )
        rerank_index=num_local_matches.cpu().numpy().argsort()[::-1]
        rerank_results.append(pre[index][rerank_index])
    
    # 计算recall@k
    rerank_results=np.array(rerank_results)
    recalls = np.zeros(len(k_values))

    for query_index, pred in enumerate(rerank_results):
        for i, n in enumerate(k_values):
            if np.any(np.in1d(pred[:n], ground_truth[query_index])):
                recalls[i:] += 1
                break

    recalls = recalls / len(rerank_results)
    recalls = {k: v for (k, v) in zip(k_values, recalls)}

    if print_results:
        print()  # print a new line
        table = PrettyTable()
        table.field_names = ["K"] + [str(k) for k in k_values]
        table.add_row(["Recall@K"] + [f"{100 * v:.2f}" for v in recalls])
        print(table.get_string(title=f"Performances rerank on {dataset_name}"))

    return recalls, rerank_results
