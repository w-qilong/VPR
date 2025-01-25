import os
import yaml

def load_checkpoint_config(ckpt_path):
    """加载checkpoint对应的配置文件"""
    if ckpt_path is None:
        return None
    # 获取version目录路径
    version_dir = os.path.dirname(os.path.dirname(ckpt_path))
    # 构建config文件路径
    config_path = os.path.join(version_dir, "hparams.yaml")
    
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        return config
    return None