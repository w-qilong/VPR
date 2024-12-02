import torch.nn.functional as F
from pytorch_metric_learning import losses, miners
from pytorch_metric_learning.distances import CosineSimilarity, DotProductSimilarity
from torch import nn
from torch.autograd import Variable
from torch.nn import CosineEmbeddingLoss


def get_loss(loss_name):
    if loss_name == 'SupConLoss': return losses.SupConLoss(temperature=0.07)
    if loss_name == 'CircleLoss': return losses.CircleLoss(m=0.4, gamma=80)  # these are params for image retrieval
    if loss_name == 'MultiSimilarityLoss': return losses.MultiSimilarityLoss(alpha=1.0, beta=50, base=0.0,
                                                                             distance=CosineSimilarity())
    if loss_name == 'ContrastiveLoss': return losses.ContrastiveLoss(pos_margin=0, neg_margin=1)
    if loss_name == 'Lifted': return losses.GeneralizedLiftedStructureLoss(neg_margin=0, pos_margin=1,
                                                                           distance=DotProductSimilarity())
    if loss_name == 'FastAPLoss': return losses.FastAPLoss(num_bins=30)

    if loss_name == 'NTXentLoss': return losses.NTXentLoss(
        temperature=0.07)  # The MoCo paper uses 0.07, while SimCLR uses 0.5.
    if loss_name == 'TripletMarginLoss':
        return losses.TripletMarginLoss(margin=0.1, swap=False, smooth_loss=False,
                                        triplets_per_anchor='all')  # or an int, for example 100
        # return nn.TripletMarginLoss(margin=0.1, p=2, reduction='mean')

    if loss_name == 'CentroidTripletLoss': return losses.CentroidTripletLoss(margin=0.05,
                                                                             swap=False,
                                                                             smooth_loss=False,
                                                                             triplets_per_anchor="all")
    
    if loss_name == 'CrossEntropy': return nn.CrossEntropyLoss(reduction='mean')

    if loss_name == 'BCEWithLogitsLoss': return nn.BCEWithLogitsLoss()

    if loss_name == 'FocalLoss': return FocalLoss(gamma=2, alpha=0.5, size_average=True)

    if loss_name == 'CosineEmbeddingLoss': return CosineEmbeddingLoss()

    raise NotImplementedError(f'Sorry, <{loss_name}> loss function is not implemented!')


def get_miner(miner_name, margin=0.1):
    # Triplet miners output a tuple of size 3: (anchors, positives, negatives).
    if miner_name == 'TripletMarginMiner': return miners.TripletMarginMiner(margin=margin,
                                                                            distance=CosineSimilarity(),
                                                                            type_of_triplets="hard")  # all, hard, semihard, easy
    # MultiSimilarityMiner outputs two tuples: (anchor, positive), (anchor negative)
    if miner_name == 'MultiSimilarityMiner': return miners.MultiSimilarityMiner(epsilon=margin,
                                                                                distance=CosineSimilarity())

    if miner_name == 'PairMarginMiner': return miners.PairMarginMiner(pos_margin=0.7, neg_margin=0.3,
                                                                      distance=DotProductSimilarity())
    if miner_name == 'BatchHardMiner': return miners.BatchHardMiner(distance=CosineSimilarity())

    if miner_name == 'BatchEasyHardMiner':
        return miners.BatchEasyHardMiner(
            distance=CosineSimilarity(),
        )
    return None


class AutomaticWeightedLoss(nn.Module):
    """automatically weighted multi-task loss

    Params：
        num: int，the number of loss
        x: multi-task loss
    Examples：
        loss1=1
        loss2=2
        awl = AutomaticWeightedLoss(2)
        loss_sum = awl(loss1, loss2)
    """

    def __init__(self, num=2):
        super(AutomaticWeightedLoss, self).__init__()
        param = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(param)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum


class dual_softmax_loss(nn.Module):
    def __init__(self, ):
        super(dual_softmax_loss, self).__init__()

    def forward(self, sim_matrix, temp=1000):
        sim_matrix = sim_matrix * F.softmax(sim_matrix / temp, dim=0) * len(
            sim_matrix)  # With an appropriate temperature parameter, the model achieves higher performance
        logpt = F.log_softmax(sim_matrix, dim=-1)
        logpt = torch.diag(logpt)
        loss = -logpt
        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, alpha=0.75, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)): self.alpha = torch.Tensor([alpha, 1 - alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1 - pt) ** self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()


import math
import torch
from torch import nn
from scipy.special import binom


class LSoftmaxLinear(nn.Module):
    def __init__(self, input_features, output_features, margin):
        super().__init__()
        self.input_dim = input_features  # number of input feature i.e. output of the last fc layer
        self.output_dim = output_features  # number of output = class numbers
        self.margin = margin  # m
        self.beta = 100
        self.beta_min = 0
        self.scale = 0.99

        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu") # gpu or cpu

        # Initialize L-Softmax parameters
        self.weight = nn.Parameter(torch.FloatTensor(input_features, output_features))
        self.divisor = math.pi / self.margin  # pi/m
        self.C_m_2n = torch.Tensor(binom(margin, range(0, margin + 1, 2))).to(device)  # C_m{2n}
        self.cos_powers = torch.Tensor(range(self.margin, -1, -2)).to(device)  # m - 2n
        self.sin2_powers = torch.Tensor(range(len(self.cos_powers))).to(device)  # n
        self.signs = torch.ones(margin // 2 + 1).to(device)  # 1, -1, 1, -1, ...
        self.signs[1::2] = -1

    def calculate_cos_m_theta(self, cos_theta):
        sin2_theta = 1 - cos_theta ** 2
        cos_terms = cos_theta.unsqueeze(1) ** self.cos_powers.unsqueeze(0)  # cos^{m - 2n}
        sin2_terms = (sin2_theta.unsqueeze(1)  # sin2^{n}
                      ** self.sin2_powers.unsqueeze(0))

        cos_m_theta = (self.signs.unsqueeze(0) *  # -1^{n} * C_m{2n} * cos^{m - 2n} * sin2^{n}
                       self.C_m_2n.unsqueeze(0) *
                       cos_terms *
                       sin2_terms).sum(1)  # summation of all terms

        return cos_m_theta

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.weight.data.t())

    def find_k(self, cos):
        # to account for acos numerical errors
        eps = 1e-7
        cos = torch.clamp(cos, -1 + eps, 1 - eps)
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            x, w = input, self.weight
            beta = max(self.beta, self.beta_min)
            logit = x.mm(w)
            indexes = range(logit.size(0))
            logit_target = logit[indexes, target]

            # cos(theta) = w * x / ||w||*||x||
            w_target_norm = w[:, target].norm(p=2, dim=0)
            x_norm = x.norm(p=2, dim=1)
            cos_theta_target = logit_target / (w_target_norm * x_norm + 1e-10)

            # equation 7
            cos_m_theta_target = self.calculate_cos_m_theta(cos_theta_target)

            # find k in equation 6
            k = self.find_k(cos_theta_target)

            # f_y_i
            logit_target_updated = (w_target_norm *
                                    x_norm *
                                    (((-1) ** k * cos_m_theta_target) - 2 * k))
            logit_target_updated_beta = (logit_target_updated + beta * logit[indexes, target]) / (1 + beta)

            logit[indexes, target] = logit_target_updated_beta
            self.beta *= self.scale
            return logit
        else:
            assert target is None
            return input.mm(self.weight)


if __name__ == '__main__':
    # awl = AutomaticWeightedLoss(2)
    # print(awl.parameters())

    losser = LSoftmaxLinear(input_features=768, output_features=2, margin=5).cuda()
    a = torch.randn((2, 768)).cuda()
    b = torch.tensor([1, 0]).cuda()
    print(losser(a, b))
