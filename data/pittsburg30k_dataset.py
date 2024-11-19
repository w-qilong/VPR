from os.path import join, exists
from collections import namedtuple
from scipy.io import loadmat

import torchvision.transforms as T
import torch.utils.data as data

from PIL import Image
from sklearn.neighbors import NearestNeighbors

root_dir = '/media/cartolab3/DataDisk/wuqilong_file/VPR_datasets/Pittsburgh_250k'

if not exists(root_dir):
    raise FileNotFoundError(
        'root_dir is hardcoded, please adjust to point to Pittsburgh dataset')

struct_dir = join(root_dir, 'datasets/')
queries_dir = join(root_dir, 'queries_real')

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
                                   'dbImage', 'utmDb', 'qImage', 'utmQ', 'numDb', 'numQ',
                                   'posDistThr', 'posDistSqThr', 'nonTrivPosDistSqThr'])

def parse_dbStruct(path):
    mat = loadmat(path)
    matStruct = mat['dbStruct'].item()

    if '250k' in path.split('/')[-1]:
        dataset = 'pitts250k'
    else:
        dataset = 'pitts30k'

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


class Pittsburg30kDataset(data.Dataset):
    def __init__(self, input_transform=None, onlyDB=False):
        super().__init__()

        self.input_transform = input_transform
        self.structFile = join(struct_dir, 'pitts30k_test.mat')

        self.dbStruct = parse_dbStruct(self.structFile)
        self.images = [join(root_dir, dbIm) for dbIm in self.dbStruct.dbImage]
        if not onlyDB:
            self.images += [join(queries_dir, qIm)
                            for qIm in self.dbStruct.qImage]

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


if __name__ == '__main__':
    def input_transform(image_size=None):
        return T.Compose([
            T.Resize(image_size),  # interpolation=T.InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


    strctFile = Pittsburg30kDataset(input_transform=input_transform(224))
    print(len(strctFile))
    for item, index in strctFile:
        print(item.shape)
        print(index)
        break
