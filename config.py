import torch
import numpy as np

device = torch.device('cuda')
batch_size = 16
initial_learning_rate = 0.0007
num_proposal = 8
size_template = torch.from_numpy(np.array([1.5, 1.6, 3.7])).to(device)
data_points = 1000
epoch_num = 50
seed_features_dim = 128
PATH = 'F:\\data_object_velodyne'
SAVE_PATH = PATH + '\\model.pth'
