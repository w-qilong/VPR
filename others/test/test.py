from pytorch_metric_learning.losses import MultiSimilarityLoss, CrossBatchMemory
from pytorch_metric_learning.miners import MultiSimilarityMiner

loss_func = MultiSimilarityLoss(alpha=0.5, beta=1.0, use_weight=True)
miner = MultiSimilarityMiner(epsilon=0.1)


loss_func = CrossBatchMemory(loss_func, embedding_size=1024, memory_size=1024, miner=miner, decay_lambda=0.05)

import torch

# 生成一些示例数据
batch_size = 256
embedding_dim = 1024
num_classes = 10

for i in range(10):  # 循环5次
    # 生成随机嵌入向量
    embeddings = torch.randn(batch_size, embedding_dim)
    
    # 生成随机标签
    labels = torch.randint(0, num_classes, (batch_size,))
    
    # 计算损失
    loss = loss_func(embeddings, labels)
    # print(f"Iteration {i}, Loss: {loss}")
