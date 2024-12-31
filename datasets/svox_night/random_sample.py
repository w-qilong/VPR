import shutil
import numpy as np
import os

def random_sample(root_path, output_path, dbImages, qImages, qIdx, pIdx, num_pairs):
    # 加载数据
    dbImages = np.load(dbImages)
    print(dbImages[0])
    qImages = np.load(qImages)
    qIdx = np.load(qIdx)
    pIdx = np.load(pIdx, allow_pickle=True)

    # 随机选择num_pairs对查询图像和参考图像
    random_indices = np.random.choice(len(qIdx), num_pairs, replace=False) # 不重复采样
    sampled_qIdx = qIdx[random_indices]
    sampled_qImages = qImages[sampled_qIdx]

    sampled_pIdx = [pIdx[i] for i in random_indices]
    sampled_dbImages = [dbImages[i] for i in sampled_pIdx]

    # 遍历保存查询和其对应的参考图像
    for i in range(num_pairs):
        query_image = os.path.join(root_path, sampled_qImages[i])
        reference_images = [os.path.join(root_path, img) for img in sampled_dbImages[i]]
        print(query_image)
        print(reference_images)
        
        # 以i为文件名保存查询和其对应的参考图像
        os.makedirs(os.path.join(output_path), exist_ok=True)
        os.makedirs(os.path.join(output_path, str(i)), exist_ok=True)
        os.makedirs(os.path.join(output_path, str(i), "query"), exist_ok=True)
        os.makedirs(os.path.join(output_path, str(i), "ref"), exist_ok=True)

        # 保存查询图像
        shutil.copy(query_image, os.path.join(output_path, str(i), "query"))
        # 保存参考图像
        for ref_image in reference_images:
            shutil.copy(ref_image, os.path.join(output_path, str(i), "ref"))



if __name__ == "__main__":

    root_path = "/media/cartolab3/DataDisk/wuqilong_file/VPR_datasets/svox/images/test"
    output_path = "/media/cartolab3/DataDisk/wuqilong_file/Projects/RerenkVPR/sample_imgs/svox_night"
    # 获取当前文件的绝对路径
    current_path = os.path.abspath(__file__)
    # 获取当前文件的父目录
    parent_path = os.path.dirname(current_path)
    # 获取当前文件的父目录的父目录

    dbImages = os.path.join(parent_path, "svox_night_val_dbImages.npy")
    qImages = os.path.join(parent_path, "svox_night_val_qImages.npy")
    qIdx = os.path.join(parent_path, "svox_night_val_qIdx.npy")
    pIdx = os.path.join(parent_path, "svox_night_val_pIdx.npy")
    num_pairs = 30
    random_sample(root_path, output_path, dbImages, qImages, qIdx, pIdx, num_pairs)