from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

DATASET_ROOT = r'/media/cartolab3/DataDisk/wuqilong_file/VPR_datasets/amstertime/images/test'
path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception('Please make sure the path to amstertime_dataset dataset is correct')

if not path_obj.joinpath('queries'):
    raise Exception(
        f'Please make sure the directory train_val from amstertime_dataset dataset is situated in the directory {DATASET_ROOT}')

if not path_obj.joinpath('database'):
    raise Exception(
        f'Please make sure the directory train_val from amstertime_dataset dataset is situated in the directory {DATASET_ROOT}')


class AmstertimeDataset(Dataset):
    def __init__(self, input_transform=None):
        self.input_transform = input_transform

        # hard coded reference image names, this avoids the hassle of listing them at each epoch.
        self.dbImages = np.load(
            'datasets/amstertime/amstertime_val_dbImages.npy')

        # hard coded query image names.
        self.qImages = np.load('datasets/amstertime/amstertime_val_qImages.npy')

        # hard coded index of query images
        self.qIdx = np.load('datasets/amstertime/amstertime_val_qIdx.npy')

        # hard coded groundtruth (correspondence between each query and its matches)
        self.pIdx = np.load('datasets/amstertime/amstertime_val_pIdx.npy',
                            allow_pickle=True)

        # concatenate reference images then query images so that we can use only one dataloader
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))

        # we need to keep the number of references so that we can split references-queries
        # when calculating recall@K
        self.num_references = len(self.dbImages)

        # you can use follow code to show some sample for query and correspond ref
        # fig,axes=plt.subplots(1,2)
        # index=100
        # query=self.qImages[self.qIdx[index]]
        # img=Image.open(DATASET_ROOT+ query)
        # axes[0].imshow(img)
        # ref=self.dbImages[self.pIdx[index]]
        # axes[1].imshow(Image.open(DATASET_ROOT+ref[0]))
        # plt.show()

    def __getitem__(self, index):
        img = Image.open(self.images[index])
        # 如果是单通道图像，将其转换为三通道
        if img.mode == 'L':
            img = Image.merge('RGB', (img, img, img))

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset=AmstertimeDataset()
    print(np.asarray(dataset[0][0]).shape)
    print(len(dataset.qImages))
    print(len(dataset.dbImages))
