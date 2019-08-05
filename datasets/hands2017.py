import os
import numpy as np
from torch.utils.data import Dataset
import imageio


# Dataset class for loading HANDS2017 training set. No test/validation split is applied.
class HANDS2017Dataset(Dataset):
    def __init__(self, data_dir, centers_path, transform=None, max_samples=None):
        self.img_width = 640
        self.img_height = 480
        self.min_depth = 100
        self.max_depth = 1500
        self.u0 = 315.944855
        self.v0 = 245.287079
        self.fx = 475.065948
        self.fy = 475.065857
        self.n_joints = 21
        self.world_dim = 3

        if not os.path.exists(data_dir):
            raise FileNotFoundError('The specified depth directory does not exist.')
        if not os.path.exists(centers_path):
            raise FileNotFoundError('The specified centers directory does not exist')

        center_strings = np.loadtxt(centers_path, dtype=str, delimiter=' ', max_rows=max_samples)

        label_file_path = os.path.join(data_dir, 'Training_Annotation.txt')
        self.labels = np.loadtxt(label_file_path, dtype='float32', usecols=range(1, 64), max_rows=max_samples)

        depth_file_names = np.loadtxt(label_file_path, dtype=str, usecols=0, max_rows=max_samples)

        invalid_mask = center_strings[:, 0] == 'invalid'
        self.centers = center_strings[~invalid_mask].astype('float32')

        self.depth_file_paths = []
        for file_name in depth_file_names[~invalid_mask]:
            self.depth_file_paths.append(os.path.join(data_dir, 'images', file_name))

        self.transform = transform

    def __getitem__(self, idx):
        depthmap = imageio.imread(self.depth_file_paths[idx])
        depthmap[depthmap == 0.0] = self.max_depth
        points = self._depthmap2points(depthmap)
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

    def _pixel2world(self, x, y, z):
        w_x = (x - self.u0) * z / self.fx
        w_y = (y - self.v0) * z / self.fy
        w_z = z
        return w_x, w_y, w_z

    def _world2pixel(self, x, y, z):
        p_x = self.u0 + x / z * self.fx
        p_y = self.v0 + y / z * self.fy
        return p_x, p_y

    def _depthmap2points(self, image):
        x, y = np.meshgrid(np.arange(self.img_width) + 1, np.arange(self.img_height) + 1)
        points = np.zeros((self.img_height, self.img_width, 3), dtype=np.float32)
        points[:, :, 0], points[:, :, 1], points[:, :, 2] = self._pixel2world(x, y, image)
        return points

    def _points2pixels(self, points):
        pixels = np.zeros((points.shape[0], 2))
        pixels[:, 0], pixels[:, 1] = self._world2pixel(points[:, 0], points[:, 1], points[:, 2])
        return pixels