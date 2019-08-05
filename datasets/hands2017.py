import os
import numpy as np
from torch.utils.data import Dataset
import imageio

from datasets import data_utils


# Dataset class for loading HANDS2017 training set. No test/validation split is applied.
class HANDS2017Dataset(Dataset):
    def __init__(self, data_dir, centers_path, transform=None):
        self.img_width = 640
        self.img_height = 480
        self.min_depth = 100
        self.max_depth = 1500
        self.fx = 475.065948
        self.fy = 475.065857
        self.n_joints = 21
        self.world_dim = 3

        if not os.path.exists(data_dir):
            raise FileNotFoundError('The specified depth directory does not exist.')
        if not os.path.exists(centers_path):
            raise FileNotFoundError('The specified centers directory does not exist')

        center_strings = np.loadtxt(centers_path, dtype=str, delimiter=' ')

        label_file_path = os.path.join(data_dir, 'Training_Annotation.txt')
        self.labels = np.loadtxt(label_file_path, dtype='float32', usecols=range(1, 64))

        depth_file_names = np.loadtxt(label_file_path, dtype=str, usecols=0)

        invalid_mask = center_strings[:, 0] == 'invalid'
        self.centers = center_strings[~invalid_mask].astype('float32')

        self.depth_file_paths = []
        for file_name in depth_file_names[~invalid_mask]:
            self.depth_file_paths.append(os.path.join(data_dir, 'images', file_name))

        self.transform = transform

    def __getitem__(self, idx):
        depthmap = imageio.imread(self.depth_file_paths[idx])
        points = data_utils.depthmap2points(depthmap, self.fx, self.fy)
        points = points.reshape((-1, 3))

        sample = {
            'name': os.path.split(self.depth_file_paths[idx])[-1],
            'points': points,
            'labels': self.labels[idx],
            'center': self.centers[idx]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return self.centers.shape[0]

