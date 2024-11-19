from pathlib import Path
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

DATASET_ROOT = r'/media/cartolab3/DataDisk/wuqilong_file/VPR_datasets/VPR_Bench_Datasets/Nordland/'
path_obj = Path(DATASET_ROOT)
if not path_obj.exists():
    raise Exception('Please make sure the path to Nordland dataset is correct')

if not path_obj.joinpath('query'):
    raise Exception(
        f'Please make sure the directory train_val from Nordland dataset is situated in the directory {DATASET_ROOT}')

if not path_obj.joinpath('ref'):
    raise Exception(
        f'Please make sure the directory train_val from Nordland dataset is situated in the directory {DATASET_ROOT}')


class NordlandDataset(Dataset):
    def __init__(self, input_transform=None):
        self.input_transform = input_transform

        # hard coded reference image names, this avoids the hassle of listing them at each epoch.
        self.dbImages = np.load(
            'datasets/nordland/nordland_val_dbImages.npy')

        # hard coded query image names.
        self.qImages = np.load('datasets/nordland/nordland_val_qImages.npy')

        # hard coded index of query images
        self.qIdx = np.load('datasets/nordland/nordland_val_qIdx.npy')

        # hard coded groundtruth (correspondence between each query and its matches)
        self.pIdx = np.load('datasets/nordland/nordland_val_pIdx.npy',
                            allow_pickle=True)

        # concatenate reference images then query images so that we can use only one dataloader
        self.images = np.concatenate((self.dbImages, self.qImages[self.qIdx]))

        # we need to keep the number of references so that we can split references-queries
        # when calculating recall@K
        self.num_references = len(self.dbImages)

        #  ## you can use follow code to show some sample for query and correspond ref
        # fig,axes=plt.subplots(1,4)
        # query=self.qImages[self.qIdx[2]]
        # img=Image.open(DATASET_ROOT+ query)
        # axes[0].imshow(img)
        # ref=self.dbImages[self.pIdx[2]]
        # print(ref)
        # axes[1].imshow(Image.open(DATASET_ROOT+ref[0]))
        # axes[2].imshow(Image.open(DATASET_ROOT+ref[1]))
        # axes[3].imshow(Image.open(DATASET_ROOT+ref[2]))
        # # axes[1].imshow(Image.open(DATASET_ROOT+ref[3]))
        # # axes[1].imshow(Image.open(DATASET_ROOT+ref[4]))
        # plt.show()

    def __getitem__(self, index):
        img = Image.open(DATASET_ROOT + self.images[index])

        if self.input_transform:
            img = self.input_transform(img)

        return img, index

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    dataset = NordlandDataset()
    # print(dataset[0])
    # sub_images = torch.utils.data.Subset(dataset, indices=range(10))
    # print(sub_images)
    print(len(dataset.qImages))
    print(len(dataset.dbImages))
