from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os

DATASET_ROOT = r'/media/cartolab3/DataDisk/wuqilong_file/VPR_datasets/VPR_Bench_Datasets/GardensPointWalking'
path_obj = Path(DATASET_ROOT)

if not path_obj.exists():
    raise Exception('Please make sure the path to Nordland dataset is correct')

if not path_obj.joinpath('query'):
    raise Exception(
        f'Please make sure the directory train_val from Nordland dataset is situated in the directory {DATASET_ROOT}')
else:
    query_folder = path_obj.joinpath('query')

if not path_obj.joinpath('ref'):
    raise Exception(
        f'Please make sure the directory train_val from Nordland dataset is situated in the directory {DATASET_ROOT}')
else:
    ref_folder = path_obj.joinpath('ref')

if not path_obj.joinpath('ground_truth_new.npy'):
    raise Exception(
        f'Please make sure the directory train_val from Nordland dataset is situated in the directory {DATASET_ROOT}')
else:
    ground_truth_path = path_obj.joinpath('ground_truth_new.npy')
    ground_truth = np.load(ground_truth_path, allow_pickle=True)
    # print(ground_truth)



dbImages = os.listdir(ref_folder)
dbImages = sorted(dbImages, key=lambda x: int(x.split('.')[0]))
dbImages = np.array([os.path.join('ref', i) for i in dbImages])


qImages = os.listdir(query_folder)
qImages = sorted(qImages, key=lambda x: int(x.split('.')[0]))
qImages = np.array([os.path.join('query', i) for i in qImages])


qIdx = np.arange(0, len(qImages))
pIdx = [np.array(i[1]) for i in ground_truth]
images = np.concatenate((dbImages, qImages[qIdx]))



# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件的目录
current_file_dir = os.path.dirname(current_file_path)

# hard code
np.save(os.path.join(current_file_dir, 'GardensPoint_walking_val_dbImages.npy'),dbImages)
np.save(os.path.join(current_file_dir, 'GardensPoint_walking_val_qImages.npy'),qImages)
np.save(os.path.join(current_file_dir, 'GardensPoint_walking_val_qIdx.npy'),qIdx)
# np.save(os.path.join(current_file_dir, 'GardensPoint_walking_val_pIdx.npz'),*pIdx)

# numpy不能保存长度不一致的列表，所以需要将每个列表转换为numpy数组，并存储在一个numpy数组中，使用object数据类型
# 将每个列表转换为numpy数组，并存储在一个numpy数组中，使用object数据类型
irregular_array = np.empty(len(pIdx), dtype=object)
for i, sublist in enumerate(pIdx):
    irregular_array[i] = np.array(sublist)

# 保存numpy数组到文件
np.save(os.path.join(current_file_dir, 'GardensPoint_walking_val_pIdx.npy'),irregular_array)

# 加载pIdx
# 从文件中加载numpy数组
irregular_array = np.load(os.path.join(current_file_dir, 'GardensPoint_walking_val_pIdx.npy'), allow_pickle=True)

# 现在irregular_array是一个numpy数组，你可以使用索引来访问数据
# 例如，访问第一个列表
first_list = irregular_array[0]
print(first_list)  # 输出: [1 2 3]

# 访问第二个列表
second_list = irregular_array[1]
print(second_list)  # 输出: [4 5]

