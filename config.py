import torch

device = torch.device('cuda')
batch_size = 16
num_proposal = 16
epoch_num = 20
seed_features_dim = 256
PATH = 'F:\data_object_velodyne'
SAVE_PATH = PATH + '\model.pth'
