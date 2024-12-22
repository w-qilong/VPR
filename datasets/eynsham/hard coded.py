from pathlib import Path
import numpy as np
import os
from glob import glob
from natsort import natsorted
from sklearn.neighbors import NearestNeighbors

positive_dist_threshold = 25

DATASET_ROOT = r'/media/cartolab3/DataDisk/wuqilong_file/VPR_datasets/eynsham/images/test'
path_obj = Path(DATASET_ROOT)

if not path_obj.exists():
    raise Exception('Please make sure the path to st_lucia dataset is correct')

if not path_obj.joinpath('query'):
    raise Exception(
        f'Please make sure the directory train_val from eynsham dataset is situated in the directory {DATASET_ROOT}')
else:
    queries_folder = path_obj.joinpath('queries')

if not path_obj.joinpath('database'):
    raise Exception(
        f'Please make sure the directory train_val from eynsham dataset is situated in the directory {DATASET_ROOT}')
else:
    database_folder = path_obj.joinpath('database')

# 根据utm坐标，确定query和positive图像

#### Read paths and UTM coordinates for all images.
database_paths = natsorted(glob(os.path.join(database_folder, "**", "*.jpg"), recursive=True)) # 绝对路径
queries_paths = natsorted(glob(os.path.join(queries_folder, "**", "*.jpg"),  recursive=True))

# The format must be path/to/file/@utm_easting@utm_northing@...@.jpg
database_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in database_paths]).astype(float)
queries_utms = np.array([(path.split("@")[1], path.split("@")[2]) for path in queries_paths]).astype(float)


# Find positives_per_query, which are within positive_dist_threshold (default 25 meters)
knn = NearestNeighbors(n_jobs=-1)
knn.fit(database_utms)
positives_per_query = knn.radius_neighbors(queries_utms,
                                                radius=positive_dist_threshold,
                                                return_distance=False)
database_num = len(database_paths)
queries_num = len(queries_paths)

qIdx = np.arange(0, len(queries_paths))
pIdx = positives_per_query

# 获取当前文件的绝对路径
current_file_path = os.path.abspath(__file__)
# 获取当前文件的目录
current_file_dir = os.path.dirname(current_file_path)

# hard code
np.save(os.path.join(current_file_dir, 'eynsham_val_dbImages.npy'),database_paths)
np.save(os.path.join(current_file_dir, 'eynsham_val_qImages.npy'),queries_paths)
np.save(os.path.join(current_file_dir, 'eynsham_val_qIdx.npy'),qIdx)

# numpy不能保存长度不一致的列表，所以需要将每个列表转换为numpy数组，并存储在一个numpy数组中，使用object数据类型
# 将每个列表转换为numpy数组，并存储在一个numpy数组中，使用object数据类型
irregular_array = np.empty(len(pIdx), dtype=object)
for i, sublist in enumerate(pIdx):
    irregular_array[i] = np.array(sublist)

# 保存numpy数组到文件
np.save(os.path.join(current_file_dir, 'eynsham_val_pIdx.npy'),irregular_array)

# 加载pIdx
# 从文件中加载numpy数组
irregular_array = np.load(os.path.join(current_file_dir, 'eynsham_val_pIdx.npy'), allow_pickle=True)

# 现在irregular_array是一个numpy数组，你可以使用索引来访问数据
# 例如，访问第一个列表
first_list = irregular_array[0]
print(first_list)  # 输出: [1 2 3]

# 访问第二个列表
second_list = irregular_array[1]
print(second_list)  # 输出: [4 5]

