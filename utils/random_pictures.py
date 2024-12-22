import os
import random
import shutil
from pathlib import Path

def copy_random_pictures(source_dir: str, target_dir: str, num_pictures: int) -> None:
    """
    从源目录随机复制指定数量的图片到目标目录
    
    参数:
        source_dir (str): 源图片目录路径
        target_dir (str): 目标目录路径
        num_pictures (int): 需要复制的图片数量
    """
    # 确保目标目录存在
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # 获取所有图片文件
    image_extensions = ('.jpg', '.jpeg', '.png', '.gif', '.bmp')
    image_files = [
        f for f in os.listdir(source_dir) 
        if os.path.isfile(os.path.join(source_dir, f)) 
        and f.lower().endswith(image_extensions)
    ]
    
    # 确保请求的图片数量不超过可用的图片总数
    num_to_copy = min(num_pictures, len(image_files))
    
    # 随机选择图片
    selected_images = random.sample(image_files, num_to_copy)
    
    # 复制选中的图片到目标目录
    for image in selected_images:
        source_path = os.path.join(source_dir, image)
        target_path = os.path.join(target_dir, image)
        shutil.copy2(source_path, target_path)

if __name__ == "__main__":
    source_dir = "/media/cartolab3/DataDisk/wuqilong_file/VPR_datasets/nordland/images/test/queries"
    target_dir = "imgs/train_random"
    num_pictures = 100
    copy_random_pictures(source_dir, target_dir, num_pictures)
