from Network import Votenet
import torch.utils.data as torch_data
from kitty import KittyDataset
import numpy as np
import cv2
import os
import loss
from config import *
from tqdm import tqdm
from matplotlib import pyplot as plt

torch.multiprocessing.freeze_support()

test_set = KittyDataset(PATH, split='test')
test_loader = torch_data.DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=1)
model = Votenet(num_class=2, num_heading_bin=2, num_size_cluster=2, mean_size_arr=np.zeros((3, 1)),
                input_feature_dim=2)
torch.multiprocessing.freeze_support()

checkpoint = torch.load(SAVE_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
loss1 = checkpoint['loss']


def show_points(points, votes, idx, image, line):
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
            cv2.line(image, (int(z[0]), int(z[1])), (int(z1[0]), int(z1[1])), color=(0, 255, 255))
    return image


def draw_3dBox(points, idx, gt, centroid_loc, draw_gt):
    s = int(test_set.image_idx_list[idx])
    image = cv2.imread(os.path.join(test_set.image_dir, '%06d.png' % s))

    P2, R0, Tr = test_set.all_calib['P2'][idx], test_set.all_calib['R0'][idx], test_set.all_calib['Tr'][idx]

    gtt = np.ndarray((12,))
    gtt[4:7] = points[4:7]
    gtt[7:10] = points[1:4]
    # gtt[10] = 0

    edges = test_set.get_box_points(gtt, from_gt=False)
    edges[0:3, :] = edges[0:3, :] + centroid_loc[0:1].transpose()

    edges = R0 @ Tr @ edges

    edges = edges / edges[3, :]

    gtt[4:7] = test_set.find_bbox_size(torch.from_numpy(np.transpose(edges))).numpy()

    gtt[7:10] = edges[0:3, 8]
    gtt[10] = gt[10]
    if not draw_gt:
        edges = test_set.get_box_points(gtt, from_gt=False)
    else:
        edges = test_set.get_box_points(gt)
    edges = P2 @ edges

    edges = edges / edges[2, :]
    edges = edges[0:2, :]
    edges = np.transpose(edges)

    for i in range(0, 9):
        if i < 8:
            cv2.circle(image, center=(int(edges[i][0]), int(edges[i][1])), radius=10, color=[255, 255, 255],
                       thickness=cv2.FILLED)
        else:
            cv2.circle(image, center=(int(edges[i][0]), int(edges[i][1])), radius=10, color=[0, 255, 255],
                       thickness=cv2.FILLED)

    lines = np.zeros((12, 2), dtype=np.int)
    lines[0] = [2, 1]
    lines[1] = [0, 1]
    lines[2] = [4, 0]
    lines[3] = [4, 7]
    lines[4] = [7, 6]
    lines[5] = [6, 2]
    lines[6] = [5, 4]
    lines[7] = [5, 6]
    lines[8] = [5, 1]
    lines[9] = [3, 7]
    lines[10] = [3, 0]
    lines[11] = [3, 2]

    for i in range(lines.shape[0]):
        cv2.line(image, (int(edges[lines[i][0], 0]), int(edges[lines[i][0], 1])),
                 (int(edges[lines[i][1], 0]), int(edges[lines[i][1], 1])), [255, 255, 255], thickness=5)

    if not draw_gt:
        cv2.imwrite("pred_image_bbox" + str(idx) + ".png", image)
    else:
        cv2.imwrite("gt_image_bbox" + str(idx) + ".png", image)


# for debugging purpose

torch.manual_seed(35)
np.random.seed(5)
if __name__ == '__main__':
    model.to(device)
    torch.multiprocessing.freeze_support()
    center_loss_values = []
    vote_loss_values = []
    size_loss_values = []
    total_loss_values = []

    for i, data in tqdm(enumerate(test_loader, 0)):
        print(i)
        data, gt, corresponding_bbox, centroid, m = data

        data = data.to(torch.float32)

        data = data.to(device)
        gt = gt.to(device)
        centroid = centroid.to(device)
        m = m.to(device)

        initial_inds = torch.unsqueeze(torch.arange(start=0, end=data.shape[1]), 0).repeat(data.shape[0], 1)

        l1_xyz, vote_xyz, result, seed_inds = model(data, initial_inds)

        l1_xyz = l1_xyz * m[:, None, None]
        vote_xyz = vote_xyz * m[:, None, None]
        result[:, 1:7, :] = result[:, 1:7, :] * m[:, None, None]
        l1_xyz = l1_xyz + centroid[:, :, None]
        vote_xyz = vote_xyz + centroid[:, None, :]
        result[:, 1:4, :] = result[:, 1:4, :] + centroid[:, :, None]

        vote_loss = loss.compute_vote_loss(gt, l1_xyz, vote_xyz, test_set)

        center_loss, size_loss, car_prob_loss = loss.compute_box_loss(gt, corresponding_bbox, result,
                                                                      seed_inds, test_set)
        print("vote loss: ", vote_loss)
        print("center loss: ", center_loss)
        print("size loss: ", size_loss)
        # print("car prob loss: ", car_prob_loss)

        vote_loss_values.append(vote_loss.data)
        center_loss_values.append(center_loss.data)
        size_loss_values.append(size_loss.data)
        # print("angle loss: ", angle_loss)
        loss1 = vote_loss.data + center_loss.data + size_loss.data

        total_loss_values.append(loss1)

        print("total loss ", loss1)

        del center_loss
        del size_loss
        del vote_loss
        del car_prob_loss
        del result
        del vote_xyz
        del l1_xyz
        del seed_inds

    plt.plot(vote_loss_values, 'b', label='Vote Loss')
    plt.plot(center_loss_values, 'r', label='Center Loss')
    plt.plot(size_loss_values, 'g', label='Size Loss')
    plt.plot(total_loss_values, 'm', label='Total Loss')
    plt.legend(framealpha=1, frameon=True)
