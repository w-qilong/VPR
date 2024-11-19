from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import os

DATASET_ROOT = r'/media/cartolab/DataDisk/wuqilong_file/VPR_datasets/VPR_Bench_Datasets/ESSEX3IN1_dataset'
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
    print(ground_truth)



dbImages = os.listdir(ref_folder)
dbImages = sorted(dbImages, key=lambda x: int(x.split('.')[0]))
dbImages = np.array([os.path.join('ref', i) for i in dbImages])


qImages = os.listdir(query_folder)
qImages = sorted(qImages, key=lambda x: int(x.split('.')[0]))
qImages = np.array([os.path.join('query', i) for i in qImages])


qIdx = np.arange(0, len(qImages))
pIdx = [np.array(i[1]) for i in ground_truth]
images = np.concatenate((dbImages, qImages[qIdx]))


# hard code
np.save('ESSEX3IN1_val_dbImages.npy',dbImages)
np.save('ESSEX3IN1_val_qImages.npy',qImages)
np.save('ESSEX3IN1_val_qIdx.npy',qIdx)
np.save('ESSEX3IN1_val_pIdx.npy',pIdx)

