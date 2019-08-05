import os
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import argparse

from lib.solver import test_epoch
from src.v2v_model import V2VModel
from src.v2v_util import V2VVoxelization
from datasets.hands2017 import HANDS2017Dataset
from lib import lua2py


#######################################################################################
# Note,
# Run in project root direcotry(ROOT_DIR) with:
# PYTHONPATH=./ python experiments/msra-subject3/predict.py -w 'path/to/model.net' -d 'path/to/dataset'
#######################################################################################

# Some helpers
def parse_args():
    parser = argparse.ArgumentParser(description='Test V2VPoseNet on HANDS2017 dataset.')
    parser.add_argument('--weights_path', '-w', required=True, type=str, help='Path to pretrained torch7 weights.')
    parser.add_argument('--data_dir', '-d', required=True, type=str, help='Path to HANDS2017 data directory.')
    parser.add_argument('--centers_path', '-c', required=True, type=str,
                        help='Path to file containing precomputed centers.')
    parser.add_argument('--device', '-g', default='cpu', type=str, help='Identifier string of target device.')
    args = parser.parse_args()
    return args


#######################################################################################

# Configurations
print('Warning: disable cudnn for batchnorm first, or just use only cuda instead!')

args = parse_args()

device = torch.device(args.device)
dtype = torch.float
batch_size = 12
n_joints = 21
cubic_size = 250

# Transforms
voxelization = V2VVoxelization(cubic_size=cubic_size, augmentation=False)


#######################################################################################

# Model
print('==> Constructing model ..')
net = V2VModel(input_channels=1, output_channels=n_joints)
lua2py.load_lua_weights(net, args.weights_path)
net = net.to(device, dtype)

#if device.type == 'cuda':
#    torch.backends.cudnn.enabled = True
#    cudnn.benchmark = True
#    print('cudnn.enabled: ', torch.backends.cudnn.enabled)


#######################################################################################

# Predict
print('==> Testing ..')
voxelize_input = voxelization.voxelize
evaluate_joints = voxelization.evaluate


def transform(sample):
    points, refpoint = sample['points'], sample['center']
    voxels = voxelize_input(points, refpoint)
    return torch.from_numpy(voxels), torch.from_numpy(refpoint.reshape((1, -1)))


def transform_output(heatmaps, centers):
    joints = evaluate_joints(heatmaps, centers)
    return joints


class BatchResultCollector():
    def __init__(self, n_samples, transform_output):
        self.n_samples = n_samples
        self.transform_output = transform_output
        self.joints = None
        self.idx = 0
    
    def __call__(self, data_batch):
        inputs_batch, outputs_batch, extra_batch = data_batch
        outputs_batch = outputs_batch.cpu().numpy()
        center_batch = extra_batch.cpu().numpy()

        joint_batch = self.transform_output(outputs_batch, center_batch)

        if self.joints is None:
            # Initialize keypoints until dimensions available now
            self.joints = np.zeros((self.n_samples, *joint_batch.shape[1:]))

        batch_size = joint_batch.shape[0]
        self.joints[self.idx:self.idx + batch_size] = joint_batch
        self.idx += batch_size

    def get_result(self):
        return self.joints


print('Test on test dataset ..')


def save_predictions(filename, predictions):
    # Reshape one sample keypoints into one line
    predictions = predictions.reshape(predictions.shape[0], -1)
    np.savetxt(filename, predictions, fmt='%0.6f')


dataset = HANDS2017Dataset(args.data_dir, args.centers_path, transform, 100)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
result_collector = BatchResultCollector(len(dataset), transform_output)

test_epoch(net, data_loader, result_collector, device, dtype)
predictions = result_collector.get_result()
save_predictions('./test_res.txt', predictions)

print('All done ..')
