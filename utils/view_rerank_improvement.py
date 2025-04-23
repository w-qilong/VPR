import torch


pth_path = 'logs/dinov2_backbone_dinov2_large/lightning_logs/version_20/improved_examples_spedtest_dataset_0.3_0.6.pth'

# 加载pth文件
improved_examples = torch.load(pth_path)

# 打印形状
print(improved_examples)

