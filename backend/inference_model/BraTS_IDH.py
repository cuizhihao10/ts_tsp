import os
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision.transforms import transforms
import nibabel as nib

modalities = ('flair', 't1ce', 't1', 't2')


def nib_load(file_name):
    if not os.path.exists(file_name):
        print('Invalid file name, can not find the file!')

    proxy = nib.load(file_name)
    data = proxy.get_fdata()  # 旋转90度
    proxy.uncache()
    return data


class Pad_test(object):
    def __call__(self, sample):
        image = sample['image']
        image = np.pad(image, ((0, 0), (0, 0), (0, 5), (0, 0)), mode='constant')
        return {'image': image}


class ToTensor_test(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']
        image = np.ascontiguousarray(image.transpose(3, 0, 1, 2))
        image = torch.from_numpy(image).float()
        return {'image': image}


def transform_test(sample):
    trans = transforms.Compose([
        Pad_test(),
        ToTensor_test()
    ])

    return trans(sample)


class BraTS(Dataset):
    def __init__(self, root='', mode='test'):  # 此处csv_file为additional data
        paths, names = [], []
        names.append(root.strip().split('/')[-2])
        file_names = os.listdir(root)
        for file in file_names:
            paths.append(file)

        self.names = names
        self.paths = paths
        self.root = root
        print(self.names)
        print('234234', self.paths)

    def __getitem__(self, index):
        paths = self.paths
        root = self.root
        images = np.stack(
            [np.array(nib_load(root + path), dtype='float32', order='C') for path in paths],
            -1)  # [240,240,155]
        print(images.shape)
        mask = images.sum(-1) > 0
        # TODO 像素值的归一化
        for k in range(4):
            x = images[..., k]  #
            y = x[mask]

            # 0.8885
            x[mask] -= y.mean()
            x[mask] /= y.std()

            images[..., k] = x

        image = images
        sample = {'image': image}
        sample = transform_test(sample)
        return sample['image'], -1, -1, -1, -1

    def __len__(self):
        return len(self.names)
