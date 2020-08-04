import torch
from main import Votenet
import torch.utils.data as torch_data
from kitty import KittyDataset
import torch.optim as optim
import numpy as np
import cv2
import os
import loss
from config import *

torch.multiprocessing.freeze_support()

test_set = KittyDataset(PATH, split='test')
test_loader = torch_data.DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=1)
model = Votenet(num_class=2, num_heading_bin=2, num_size_cluster=2, mean_size_arr=np.zeros((3, 1)),
                input_feature_dim=2)
torch.multiprocessing.freeze_support()
optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

checkpoint = torch.load(SAVE_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
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


def draw_3dBox(points, idx, gt):
    s = int(test_set.image_idx_list[idx])
    image = cv2.imread(os.path.join(test_set.image_dir, '%06d.png' % s))

    P2, R0, Tr = test_set.all_calib['P2'][idx], test_set.all_calib['R0'][idx], test_set.all_calib['Tr'][idx]

    center = points[1:4]
    size = points[4:7]

    c = np.ones((4,1))
    s = np.ones((4,1))

    c[0:3,0] = center
    s[0:3,0] = size

    center = R0 @ Tr @ c
    center = center/center[3]
    center = center[0:3,0]

    size = R0 @ Tr @ s
    size = size / size[3]
    size = size[0:3,0]


    gt[7:10] = center
    gt[4:7] = size
    gt[10] = points[7]


    bbox = test_set.get_rect_points(gt)

    points = P2 @ bbox

    points = points / points[2, :]
    points = np.transpose(points)

    for index in range(points.shape[0]):
        z = points[index]
        cv2.circle(image, center=(int(z[0]), int(z[1])), radius=10, color=[0, 0, 255])

    centerrr = gt[7:10]
    cen = np.ones((4, 1))
    cen[0:3, 0] = centerrr
    cen = P2 @ cen
    cen = cen / cen[2]
    cv2.circle(image, center=(int(cen[0]), int(cen[1])), radius=10, color=[255, 255, 255], thickness=cv2.FILLED)

    cv2.imwrite("bbox" + str(idx) + "_" + str(test_set.temp) + ".png", image)
    test_set.temp = test_set.temp + 1

    # data = data.detach().numpy()
    # aggregation = aggregation.detach().numpy()
    # image = show_points(l1_xyz, vote_xyz[0], idx, image, False)
    # cv2.imwrite("object_points_eval" + str(idx) + ".png", image)
    # draw_3dBox(aggregation[0, :, 0], idx)


torch.manual_seed(10)
np.random.seed(0)
if __name__ == '__main__':
    model.to(device)
    torch.multiprocessing.freeze_support()

    data, gt, corresponding_bbox = next(iter(test_loader))

    data = data.to(torch.float32)
    data = data.to(device)

    initial_inds = torch.unsqueeze(torch.arange(start=0, end=data.shape[1]), 0).repeat(data.shape[0], 1)

    l1_xyz, vote_xyz, aggregation, seed_inds = model(data, initial_inds)

    sample_num = 0

    # forward + backward + optimize
    # vote_loss = loss.compute_vote_loss(gt[sample_num:sample_num + 1], l1_xyz[sample_num:sample_num + 1],
    #                                    vote_xyz[sample_num:sample_num + 1], test_set)
    # box_loss = loss.compute_box_loss(gt[sample_num:sample_num + 1], corresponding_bbox[sample_num:sample_num + 1],
    #                                  aggregation[sample_num:sample_num + 1], seed_inds[sample_num:sample_num + 1],
    #                                  test_set)
    # print("vote loss: ", vote_loss)
    # print("box loss: ", box_loss)
    # loss1 = torch.sum(vote_loss + box_loss)
    # print("loss ", loss1)

    idx = int(gt[sample_num][0][11])
    s = int(test_set.image_idx_list[idx])
    data = data.cpu().numpy()
    aggregation = aggregation.cpu().detach().numpy()
    gt = gt.cpu().detach().numpy()
    # l1_xyz = l1_xyz.cpu().detach().numpy()
    # vote_xyz = vote_xyz.cpu().detach().numpy()
    # image = cv2.imread(os.path.join(test_set.image_dir, '%06d.png' % s))
    # image = show_points(np.transpose(l1_xyz[sample_num]), vote_xyz[sample_num], idx, image, False)
    # cv2.imwrite("object_points_eval" + str(idx) + ".png", image)
    # draw_3dBox(aggregation[sample_num, :, 0], idx, gt[0][0])
    # draw_3dBox(aggregation[sample_num, :, 1], idx, gt[0][0])
    # draw_3dBox(aggregation[sample_num, :, 2], idx, gt[0][0])
    # draw_3dBox(aggregation[sample_num, :, 3], idx, gt[0][0])
    # draw_3dBox(aggregation[sample_num, :, 4], idx, gt[0][0])
    # draw_3dBox(aggregation[sample_num, :, 5], idx, gt[0][0])
    # draw_3dBox(aggregation[sample_num, :, 6], idx, gt[0][0])
    # draw_3dBox(aggregation[sample_num, :, 7], idx, gt[0][0])
