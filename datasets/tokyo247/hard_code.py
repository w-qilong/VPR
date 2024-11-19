import os.path
from collections import namedtuple
from os.path import join
from pathlib import Path
from os.path import join, exists
import torch.utils.data as data
from PIL import Image
from scipy.io import loadmat
from sklearn.neighbors import NearestNeighbors

#
# DATASET_ROOT = r'/media/cartolab/DataDisk/wuqilong_file/VPR_datasets/Tokyo247'
# path_obj = Path(DATASET_ROOT)
#
# if not path_obj.exists():
#     raise Exception('Please make sure the path to Tokyo247 dataset is correct')
#
# ground_truth_path = path_obj.joinpath('datasets/tokyo247.mat')
# ground_truth = loadmat(ground_truth_path)
# # print(ground_truth)
# query_folder = path_obj.joinpath('Tokyo247query_subset_v2')
# matStruct = ground_truth['dbStruct'].item()
# whichSet = matStruct[0].item()
# # all test images path in rootpath
# dbImage = [f[0].item() for f in matStruct[1]]
# print(len(dbImage))
# # 每张图像的utm位置坐标
# utmDb = matStruct[2].T
# # 测试时所用的查询图像, 查询所用图像都不在dbImage中
# qImage = [f[0].item() for f in matStruct[3]]
# # 查询图像的utm坐标
# utmQ = matStruct[4].T
# # 数据库和查询图像的数量
# numDb = matStruct[5].item()
# numQ = matStruct[6].item()
#
# # 两个样本互为正样本的阈值距离：25
# posDistThr = matStruct[7].item()
# posDistSqThr = matStruct[8].item()
# nonTrivPosDistSqThr = matStruct[9].item()
#
# dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
#                                    'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
#                                    'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])
#
# db = dbStruct(whichSet, 'tokyo247', dbImage, utmDb, qImage,
#               utmQ, numDb, numQ, posDistThr,
#               posDistSqThr, nonTrivPosDistSqThr)
# print(len(db[3]))


root_dir = '/media/cartolab/DataDisk/wuqilong_file/VPR_datasets/Tokyo247'
if not exists(root_dir):
    raise FileNotFoundError(
        'root_dir is hardcoded, please adjust to point to Tokyo247 dataset')

struct_dir = join(root_dir, 'datasets')
queries_dir = join(root_dir, 'Tokyo247query_subset_v2')

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
                                   'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
                                   'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])


def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()
    dataset = 'tokyo247'
    whichSet = matStruct[0].item()
    # all test images path in rootpath
    dbImage = [f[0].item() for f in matStruct[1]]
    # 每张图像的utm位置坐标
    utmDb = matStruct[2].T
    # 测试时所用的查询图像, 查询所用图像都不在dbImage中
    qImage = [f[0].item() for f in matStruct[3]]
    # 查询图像的utm坐标
    utmQ = matStruct[4].T
    # 数据库和查询图像的数量
    numDb = matStruct[5].item()
    numQ = matStruct[6].item()

    # 两个样本互为正样本的阈值距离：25
    posDistThr = matStruct[7].item()
    posDistSqThr = matStruct[8].item()
    nonTrivPosDistSqThr = matStruct[9].item()

    return dbStruct(whichSet, dataset, dbImage, utmDb, qImage,
                    utmQ, numDb, numQ, posDistThr,
                    posDistSqThr, nonTrivPosDistSqThr)


class Tokyo247Dataset(data.Dataset):
    def __init__(self, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform
        self.structFile = join(struct_dir, 'tokyo247.mat')

        self.dbStruct = parse_dbStruct(self.structFile)
        self.images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        self.images = [i.replace('.jpg', '.png') for i in self.images]
        if not onlyDB:
            self.images += [join(queries_dir, qIm)
                            for qIm in self.dbStruct.qImage]
        # self.images=[i for i in self.images if os.path.exists(i)]
        print(len(self.images))

        self.whichSet = self.dbStruct.whichSet
        self.dataset = self.dbStruct.dataset

        self.positives = None
        self.distances = None

    def __getitem__(self, index):
        img = Image.open(self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)

    def getPositives(self):
        # positives for evaluation are those within trivial threshold range
        # fit NN to find them, search by radius
        if self.positives is None:
            knn = NearestNeighbors(n_jobs=-1)
            knn.fit(self.dbStruct.utmDb)

            self.distances, self.positives = knn.radius_neighbors(self.dbStruct.utmQ,
                                                                  radius=self.dbStruct.posDistThr)

        return self.positives


strctFile = Tokyo247Dataset(onlyDB=False)
for item, index in strctFile:
    print(item)
    break
