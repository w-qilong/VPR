from ..distances import CosineSimilarity
from ..utils import common_functions as c_f
from ..utils import loss_and_miner_utils as lmu
from .generic_pair_loss import GenericPairLoss
import torch


class MultiSimilarityLoss(GenericPairLoss):
    """
    modified from https://github.com/MalongTech/research-ms-loss/
    Args:
        alpha: The exponential weight for positive pairs
        beta: The exponential weight for negative pairs
        base: The shift in the exponent applied to both positive and negative pairs
    """

    def __init__(self, alpha=2, beta=50, base=0.5, **kwargs):
        super().__init__(mat_based_loss=True, **kwargs)
        self.alpha = alpha
        self.beta = beta
        self.base = base
        self.add_to_recordable_attributes(
            list_of_names=["alpha", "beta", "base"], is_stat=False
        )
        self.weight = None

    def _compute_loss(self, mat, pos_mask, neg_mask):
        pos_exp = self.distance.margin(mat, self.base)
        neg_exp = self.distance.margin(self.base, mat)

        pos_loss = (1.0 / self.alpha) * lmu.logsumexp(
            self.alpha * pos_exp, keep_mask=pos_mask.bool(), add_one=True
        )

        # 如果weight为None，则不使用权重
        if self.weight is None:
            neg_loss = (1.0 / self.beta) * lmu.logsumexp(
                self.beta * neg_exp, keep_mask=neg_mask.bool(), add_one=True
            )
        else:
            # print("Using weight in MultiSimilarityLoss")
            # 如果weight不为None，则使用权重
            weight = self.weight
            exp_term = torch.exp(self.beta * neg_exp)
            # 应用权重和掩码，权重会通过广播机制应用到每一行
            weighted_exp = exp_term * weight * neg_mask
            # 计算加权指数和，对每行求和并加1
            row_sum = torch.sum(weighted_exp, dim=1, keepdim=True) + 1.0
            # 应用log和缩放
            neg_loss = (1.0 / self.beta) * torch.log(row_sum)

        return {
            "loss": {
                "losses": pos_loss + neg_loss,
                "indices": c_f.torch_arange_from_size(mat),
                "reduction_type": "element",
            }
        }

    def get_default_distance(self):
        return CosineSimilarity()
