import torch
from main import Votenet
import torch.utils.data as torch_data
from kitty import KittyDataset
import torch.optim as optim
import numpy as np
import cv2
import os

PATH = 'F:\data_object_velodyne'
SAVE_PATH = PATH + '\model.pth'

torch.multiprocessing.freeze_support()

test_set = KittyDataset(PATH,split='test')
train_loader = torch_data.DataLoader(test_set, shuffle=True, batch_size=1, num_workers=1)
model = Votenet(num_class=2, num_heading_bin=2, num_size_cluster=2, mean_size_arr=np.zeros((3, 1)),
                input_feature_dim=2)
torch.multiprocessing.freeze_support()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

checkpoint = torch.load(SAVE_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
epoch = checkpoint['epoch']
loss = checkpoint['loss']

def show_points(points, votes, idx, image,line):
    P2, R0, Tr = test_set.all_calib['P2'][idx], test_set.all_calib['R0'][idx], test_set.all_calib['Tr'][idx]
    y = np.ones((points.shape[0], 4))
    y[:, 0:3] = points
    y = np.transpose(y)

    y = R0 @ Tr @ y
    y = y / y[3, :]
    y = np.transpose(y)
    y = y[:, 0:3]
    for index in range(y.shape[0]):
        x = np.ones((4,))
        x[0:3] = y[index]
        z = P2 @ x
        z = z / z[2]
        cv2.circle(image, center=(int(z[0]), int(z[1])), radius=1, color=[255, 255, 255])

    y1 = np.ones((votes.shape[0], 4))
    y1[:, 0:3] = votes
    y1 = np.transpose(y1)

    y1 = R0 @ Tr @ y1
    y1 = y1 / y1[3, :]
    y1 = np.transpose(y1)
    y1 = y1[:, 0:3]

    for index in range(y1.shape[0]):
        x = np.ones((4,))
        x[0:3] = y1[index]
        z = P2 @ x
        z = z / z[2]
        cv2.circle(image, center=(int(z[0]), int(z[1])), radius=1, color=[0, 0, 255])
        if line:
            x1 = np.ones((4,))
            x1[0:3] = y[index]
            z1 = P2 @ x1
            z1 = z1 / z1[2]
            cv2.line(image,(int(z[0]),int(z[1])),(int(z1[0]),int(z1[1])),color=(0,255,255))
    return image


if __name__ == '__main__':
    data, gt = test_set.__getitem__(122)
    data = torch.from_numpy(np.expand_dims(data, [0]))
    gt = torch.from_numpy(np.expand_dims(gt, [0]))
    idx = int(gt[0][0][11])
    data = data.to(torch.float32)
    s = int(test_set.image_idx_list[idx])
    image = cv2.imread(os.path.join(test_set.image_dir, '%06d.png' % s))

    l1_xyz, vote_xyz, aggregation = model(data)
    l1_xyz = l1_xyz.detach().numpy()
    l1_xyz = np.transpose(l1_xyz[0])
    vote_xyz = vote_xyz.detach().numpy()
    data = data.detach().numpy()
    # image = show_points(data[0],vote_xyz[0], idx,image,False)
    image = show_points(l1_xyz, vote_xyz[0], idx, image,False)
    cv2.imwrite("object_points_eval" + str(idx) + ".png", image)
    # cv2.imwrite("object_points" + str(idx)+ ".png", image)
