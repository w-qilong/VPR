from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import (
    CosineSimilarity,
    DotProductSimilarity,
    LpDistance,
)
from pytorch_metric_learning.losses import CrossBatchMemory
from torch import nn


class MetricLoss(nn.Module):
    """度量学习损失函数类

    实现了多重相似度损失(MultiSimilarity)和三元组损失(Triplet)两种度量学习方法
    """

    def __init__(self, loss_name, margin=0.1):
        """
        参数:
            loss_name: 损失函数名称 ('MultiSimilarityLoss' 或 'TripletMarginLoss')
            margin: 边界值,用于控制相似/不相似样本间的距离
        """
        super(MetricLoss, self).__init__()

        self.loss_name = loss_name
        self.margin = margin

        if self.loss_name == "MultiSimilarityLoss":
            # 多重相似度损失
            # alpha和beta控制正负样本的权重
            # distance使用余弦相似度计算样本间距离
            # 默认使用mean reducer
            self.loss_fn = losses.MultiSimilarityLoss(
                alpha=1.0, beta=50, base=0.0, distance=CosineSimilarity()
            )
            # 多重相似度采样器,用于选择有意义的样本对
            self.miner = miners.MultiSimilarityMiner(
                epsilon=margin, distance=CosineSimilarity()
            )
            
        elif self.loss_name == "TripletMarginLoss":
            # 三元组损失
            # margin控制锚点样本与正负样本之间的距离差
            self.loss_fn = losses.TripletMarginLoss(
                margin=margin, swap=False, smooth_loss=False, triplets_per_anchor="all"
            )
            # 硬三元组采样器,选择最难的三元组样本
            self.miner = miners.TripletMarginMiner(
                margin=margin, distance=CosineSimilarity(), type_of_triplets="hard"
            )

        elif self.loss_name == "ContrastiveLoss":
            self.loss_fn = losses.ContrastiveLoss(distance=CosineSimilarity())
            self.miner = miners.MultiSimilarityMiner(
                epsilon=margin, distance=CosineSimilarity()
            )

        elif self.loss_name == "NCALoss":
            self.loss_fn = losses.NCALoss()
            self.miner = None

    def forward(self, cls_token, labels):
        """前向传播函数

        参数:
            cls_token: 模型输出的特征向量
            labels: 样本标签
        返回:
            计算得到的损失值
        """

        # 根据损失函数类型计算相应的损失
        if self.loss_name == "MultiSimilarityLoss":
            miner_outputs = self.miner(cls_token, labels)
            return self.loss_fn(cls_token, labels, miner_outputs), miner_outputs
        elif self.loss_name == "TripletMarginLoss":
            miner_outputs = self.miner(cls_token, labels)
            return self.loss_fn(cls_token, labels, miner_outputs), miner_outputs
        elif self.loss_name == "ContrastiveLoss":
            miner_outputs = self.miner(cls_token, labels)
            return self.loss_fn(cls_token, labels, miner_outputs), miner_outputs
        elif self.loss_name == "NCALoss":
            return self.loss_fn(cls_token, labels), None
