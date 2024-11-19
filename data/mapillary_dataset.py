from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# NOTE: you need to download the mapillary_sls dataset from  https://github.com/FrederikWarburg/mapillary_sls
# make sure the path where the mapillary_sls validation dataset resides on your computer is correct.
# the folder named train_val should reside in DATASET_ROOT path (that's the only folder you need from mapillary_sls)
# I hardcoded the groundtruth for image to image evaluation, otherwise it would take ages to run the groundtruth script at each epoch.
DATASET_ROOT = r'/media/cartolab3/DataDisk/wuqilong_file/VPR_datasets/MSLS_Dataset/MSLS/'

path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception('Please make sure the path to mapillary_sls dataset is correct')

if not path_obj.joinpath('train_val'):
    raise Exception(
        f'Please make sure the directory train_val from mapillary_sls dataset is situated in the directory {DATASET_ROOT}')


class MapillaryDataset(Dataset):
    def __init__(self, input_transform=None):
        self.input_transform = input_transform

        # hard coded reference image names, this avoids the hassle of listing them at each epoch.
        self.dbImages = np.load(
            'datasets/msls_val/msls_val_dbImages.npy',
            allow_pickle=True)

        # hard coded query image names.
        self.qImages = np.load('datasets/msls_val/msls_val_qImages.npy',
                               allow_pickle=True)

        # hard coded index of query images
        self.qIdx = np.load('datasets/msls_val/msls_val_qIdx.npy',
                            allow_pickle=True)

        # hard coded groundtruth (correspondence between each query and its matches)
        self.pIdx = np.load('datasets/msls_val/msls_val_pIdx.npy',
                            allow_pickle=True)

        # concatenate reference images then query images so that we can use only one dataloader
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))

        # we need to keep the number of references so that we can split references-queries
        # when calculating recall@K
        self.num_references = len(self.dbImages)

        # you can use follow code to show some sample for query and correspond ref
        # fig, axes = plt.subplots(2, 3)
        # index = 450
        # query = self.qImages[self.qIdx[index]]
        # img = Image.open(DATASET_ROOT + query)
        # print(DATASET_ROOT + query)
        # axes[0][0].imshow(img)
        # ref = self.dbImages[self.pIdx[index]]
        # print([DATASET_ROOT + i for i in ref])
        # axes[0][1].imshow(Image.open(DATASET_ROOT + ref[0]))
        # axes[0][2].imshow(Image.open(DATASET_ROOT + ref[1]))
        # axes[1][0].imshow(Image.open(DATASET_ROOT + ref[2]))
        # axes[1][1].imshow(Image.open(DATASET_ROOT + ref[3]))
        # axes[1][2].imshow(Image.open(DATASET_ROOT + ref[4]))
        # plt.show()

    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT+self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = MapillaryDataset()
    print(f'len of dbImages:{len(dataset.dbImages)}')
    print(f'len of qImages:{len(dataset.qImages)}')
    print(f'len of qIdx:{len(dataset.qIdx)}')
    print(f'len of pIdx:{len(dataset.pIdx)}')
    # print(np.unique(dataset.qIdx))
