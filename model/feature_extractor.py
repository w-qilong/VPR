import torch
from torch import nn
from backbones import DinoV2Backbone,ResNetFPN_8_2,ResNetFPN_16_4

# 为细粒度CNN特征提取配置参数
configs = {
        'ResNetFPN_8_2': {
            'initial_dim': 128,
            'block_dims': [128, 196, 256]
        },
        'ResNetFPN_16_4': {
            'initial_dim': 128,
            'block_dims': [128, 196, 256, 384]
        }
    }

# 特征提取器
class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.corse_feature_extractor = DinoV2Backbone(freeze_layers=True)

    def forward(self,x):
        # 粗粒度特征
        cls_token,patch_tokens = self.corse_feature_extractor(x)
        return cls_token,patch_tokens

if __name__ == "__main__":
    model = FeatureExtractor()
    x=torch.randn(2,3,224,224)
    x=model(x)
    # 粗粒度特征
    print(x[0].shape)
    # 细粒度特征
    print(x[1].shape)