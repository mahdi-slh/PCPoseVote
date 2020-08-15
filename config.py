import torch

device = torch.device('cuda')
batch_size = 16
initial_learning_rate = 0.0007
num_proposal = 8
epoch_num = 20
seed_features_dim = 256
PATH = 'F:\\data_object_velodyne'
SAVE_PATH = PATH + '\\model.pth'
