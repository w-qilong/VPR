from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os

DATASET_ROOT = r'/media/cartolab/DataDisk/wuqilong_file/VPR_datasets/VPR_Bench_Datasets/Nordland'
# path_obj = Path(DATASET_ROOT)

# if not path_obj.exists():
#     raise Exception('Please make sure the path to Nordland dataset is correct')

# if not path_obj.joinpath('query'):
#     raise Exception(
#         f'Please make sure the directory train_val from Nordland dataset is situated in the directory {DATASET_ROOT}')
# else:
#     query_folder = path_obj.joinpath('query')

# if not path_obj.joinpath('ref'):
#     raise Exception(
#         f'Please make sure the directory train_val from Nordland dataset is situated in the directory {DATASET_ROOT}')
# else:
#     ref_folder = path_obj.joinpath('ref')

# if not path_obj.joinpath('ground_truth_new.npy'):
#     raise Exception(
#         f'Please make sure the directory train_val from Nordland dataset is situated in the directory {DATASET_ROOT}')
# else:
#     ground_truth_path = path_obj.joinpath('ground_truth_new.npy')
#     ground_truth = np.load(ground_truth_path, allow_pickle=True)
#     print(ground_truth[2])

#     ' ground truth: [2757 list([27569, 27570, 27571])]'

# dbImages = os.listdir(ref_folder)
# dbImages = sorted(dbImages, key=lambda x: int(x.split('.')[0]))
# dbImages = np.array([os.path.join('ref', i) for i in dbImages])
# print(len(dbImages))


# qImages = os.listdir(query_folder)
# qImages = sorted(qImages, key=lambda x: int(x.split('.')[0]))
# qImages = np.array([os.path.join('query', i) for i in qImages])
# print(len(qImages))


# qIdx = np.arange(0, len(qImages))
# pIdx = [np.array(i[1]) for i in ground_truth]
# images = np.concatenate((dbImages, qImages[qIdx]))


# hard code
# np.save('nordland_val_dbImages.npy',dbImages)
# np.save('nordland_val_qImages.npy',qImages)
# np.save('nordland_val_qIdx.npy',qIdx)
# np.save('nordland_val_pIdx.npy',pIdx)

# 加载
# dbImages = np.load('nordland_val_dbImages.npy')
# qImages = np.load('nordland_val_qImages.npy')
# qIdx = np.load('nordland_val_qIdx.npy')
pIdx = np.load('/media/cartolab3/DataDisk/wuqilong_file/Projects/RerenkVPR/datasets/nordland/nordland_val_pIdx.npy',allow_pickle=True)
print(pIdx[2])

