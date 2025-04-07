from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity
from torch import nn


class MetricLoss(nn.Module):
    """度量学习损失函数类
    
    实现了多种度量学习方法，包括多重相似度损失(MultiSimilarity)、三元组损失(Triplet)、
    对比损失(Contrastive)和NCA损失(NCA)
    """

    def __init__(self, loss_name, margin=0.1, use_weight_decay=False):
        """
        参数:
            loss_name: 损失函数名称 ('MultiSimilarityLoss', 'TripletMarginLoss', 'ContrastiveLoss' 或 'NCALoss')
            margin: 边界值，用于控制相似/不相似样本间的距离
            use_weight_decay: 是否使用权重衰减（仅用于MultiSimilarityLoss）
        """
        super(MetricLoss, self).__init__()

        self.loss_name = loss_name
        self.margin = margin
        self.use_weight_decay = use_weight_decay
        # 定义损失函数和采样器的配置
        loss_configs = {
            "MultiSimilarityLoss": {
                "loss": losses.MultiSimilarityLoss(
                    alpha=2.0, beta=50.0, base=0.5, 
                    distance=CosineSimilarity(), 
                    use_weight=use_weight_decay
                ),
                "miner": miners.MultiSimilarityMiner(
                    epsilon=margin, 
                    distance=CosineSimilarity()
                )
            },
            "TripletMarginLoss": {
                "loss": losses.TripletMarginLoss(
                    margin=margin, swap=False, 
                    smooth_loss=False, 
                    triplets_per_anchor="all"
                ),
                "miner": miners.TripletMarginMiner(
                    margin=margin, 
                    distance=CosineSimilarity(), 
                    type_of_triplets="hard"
                )
            },
            "ContrastiveLoss": {
                "loss": losses.ContrastiveLoss(
                    distance=CosineSimilarity()
                ),
                "miner": miners.MultiSimilarityMiner(
                    epsilon=margin, 
                    distance=CosineSimilarity()
                )
            },
            "NCALoss": {
                "loss": losses.NCALoss(),
                "miner": None
            }
        }
        
        # 根据损失函数名称设置相应的损失函数和采样器
        if loss_name in loss_configs:
            self.loss_fn = loss_configs[loss_name]["loss"]
            self.miner = loss_configs[loss_name]["miner"]
        else:
            raise ValueError(f"不支持的损失函数: {loss_name}")

    def forward(self, cls_token, labels):
        """前向传播函数

        参数:
            cls_token: 模型输出的特征向量
            labels: 样本标签
        返回:
            计算得到的损失值和采样器输出（如果有）
        """
        # 如果没有采样器，直接计算损失
        if self.miner is None:
            return self.loss_fn(cls_token, labels), None
        
        # 使用采样器选择样本，然后计算损失
        miner_outputs = self.miner(cls_token, labels)
        return self.loss_fn(cls_token, labels, miner_outputs), miner_outputs
