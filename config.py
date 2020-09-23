import torch
import numpy as np

torch.manual_seed(0)
np.random.seed(0)

FAR_THRESHOLD = 9
NEAR_THRESHOLD = 0.6
OBJECTNESS_CLS_WEIGHTS = np.array([0.8, 0.2])

device = torch.device('cuda')
batch_size = 16
initial_learning_rate = 0.0007
farthest_point_num = 40
num_proposal = 64
size_template = torch.from_numpy(np.array([1.5, 1.6, 3.7])).to(device)
data_points = 4000
sample_box_size = np.array([12.0, 3.0, 6.0])
random_shift_interval = np.array([2.0, 2.0, 2.0])
epoch_num = 50
seed_features_dim = 256
PATH = 'F:\\data_object_velodyne'
SAVE_PATH = PATH + '\\model.pth'
