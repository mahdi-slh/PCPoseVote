import os
import torch.utils.data as torch_data
import numpy as np
import cv2
import torch
import open3d as o3d
import random
from config import *


class KittyDataset(torch_data.Dataset):
    def __init__(self, root_dir, npoints=500, split='train', mode='TRAIN', random_select=True):
        self.temp = 1
        self.split = split
        is_test = self.split == 'test'
        self.imageset_dir = os.path.join(root_dir, 'training')
        self.label_dir = os.path.join(self.imageset_dir, 'label_2')
        self.image_dir = os.path.join(self.imageset_dir, 'image_2')
        self.eval_dir = os.path.join(root_dir, 'evaluation')
        self.calib_dir = os.path.join(self.imageset_dir, 'calib')
        split_dir = os.path.join(root_dir, split + '.txt')
        self.image_idx_list = [x.strip() for x in open(split_dir).readlines()]
        self.num_sample = self.image_idx_list.__len__()
        self.lidar_dir = os.path.join(self.imageset_dir, 'velodyne')
        self.class_map = {'Car': 1, 'Van': 2, 'Truck': 3, 'Pedestrian': 4, 'Person_sitting': 5, 'Cyclist': 6, 'Tram': 7,
                          'Misc': 8, 'DontCare': 9}
        self.npoints = npoints
        split_file = os.path.join(split_dir)
        split_file = os.path.abspath(split_file)
        self.image_idx_list = [x.strip() for x in open(split_file).readlines()]
        self.classes = [1]
        self.max_objects = 20

        self.all_calib = {'P2': np.ndarray((self.__len__(), 3, 4)), 'R0': np.ndarray((self.__len__(), 4, 4)),
                          'Tr': np.ndarray((self.__len__(), 4, 4))}

        for index in range(self.__len__()):
            self.read_calib(int(index))

    def read_calib(self, idx):
        s = int(self.image_idx_list[idx])
        calib = open(os.path.join(self.calib_dir, '%06d.txt' % s))
        all_lines = calib.readlines()
        p2_line = all_lines[2]
        p2_string = p2_line.split(" ")

        P2 = np.ndarray((12,))
        for i in range(12):
            P2[i] = float(p2_string[i + 1])
        P2 = P2.reshape((3, 4))
        self.all_calib['P2'][idx] = P2

        Tr_line = all_lines[5]
        Tr_string = Tr_line.split(" ")
        Tr = np.ndarray((12,))

        for i in range(12):
            Tr[i] = float(Tr_string[i + 1])
        Tr = Tr.reshape((3, 4))
        Tr_velo_to_cam = np.ndarray((4, 4))
        Tr_velo_to_cam[0:3, :] = Tr
        Tr_velo_to_cam[3, :] = [0.0, 0.0, 0.0, 1.0]
        self.all_calib['Tr'][idx] = Tr_velo_to_cam

        R0_rect = all_lines[4]
        R0_string = R0_rect.split(" ")
        temp = np.ndarray((9,))

        for i in range(9):
            temp[i] = float(R0_string[i + 1])
        R = temp.reshape((3, 3))
        R0 = np.ndarray((4, 4))
        R0[0:3, 0:3] = R
        R0[3][0:4] = [0, 0, 0, 1]
        R0[0][3] = 0.0
        R0[1][3] = 0.0
        R0[2][3] = 0.0
        self.all_calib['R0'][idx] = R0

    def get_rect_points(self, gt):

        R = np.ndarray((3, 3))
        R[0] = [np.cos(gt[10]), 0, np.sin(gt[10])]
        R[1] = [0, 1, 0]
        R[2] = [-np.sin(gt[10]), 0, np.cos(gt[10])]

        l = gt[6]
        w = gt[5]
        h = gt[4]

        corners_3d = np.ndarray((3, 9))
        corners_3d[0] = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, 0]
        corners_3d[1] = [0, 0, 0, 0, -h, -h, -h, -h, 0]
        corners_3d[2] = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, 0]

        # corners_3d = R @ corners_3d

        corners_3d[0, :] = corners_3d[0, :] + gt[7]
        corners_3d[1, :] = corners_3d[1, :] + gt[8]
        corners_3d[2, :] = corners_3d[2, :] + gt[9]

        points_3d = np.ndarray((4, 9))
        points_3d[0:3, :] = corners_3d
        points_3d[3, :] = [1] * 9

        return points_3d

    def is_equal(self, x1, y1, z1, t1, x2, y2, z2, t2):
        epsilon = 2
        x1 = x1 / t1
        y1 = y1 / t1
        z1 = z1 / t1

        x2 = x2 / t2
        y2 = y2 / t2
        z2 = z2 / t2

        dist = (x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2

        return dist < epsilon

    def draw_3dBox(self, gt):
        idx = int(gt[11])
        s = int(self.image_idx_list[idx])
        image = cv2.imread(os.path.join(self.image_dir, '%06d.png' % s))

        points_3d = self.get_rect_points(gt)

        P2, _, _ = self.all_calib['P2'][idx], self.all_calib['R0'][idx], self.all_calib['Tr'][idx]

        pts_2D = np.matmul(P2, points_3d)
        pts_2D[0, :] = pts_2D[0, :] / pts_2D[2, :]
        pts_2D[1, :] = pts_2D[1, :] / pts_2D[2, :]

        pts_2D = pts_2D[0:2, :]

        for i in range(0, 8):
            # cv2.line(image, (int(pts_2D[0][i]), int(pts_2D[1][i])), (int(pts_2D[0][i + 1]), int(pts_2D[1][i + 1])),
            #          [255, 0, 0])
            cv2.circle(image, center=(int(pts_2D[0][i]), int(pts_2D[1][i])), radius=5, color=[255, 0, 0])

        cv2.imwrite("asas" + str(self.temp) + ".png", image)

    def __len__(self):
        return len(self.image_idx_list)

    def __getitem__(self, item):
        s = int(self.image_idx_list[item])
        path = os.path.join(self.lidar_dir, '%06d.bin' % s)
        label_path = os.path.join(self.label_dir, '%06d.txt' % s)
        lines = [x.strip() for x in open(label_path).readlines()]
        gt = -np.ones((self.max_objects, 12))
        ind = -1
        inds = []
        for i in range(len(lines)):
            r = lines[i].split(' ')
            if self.class_map[r[0]] not in self.classes:
                continue
            ind = ind + 1
            inds.append(ind)
            gt[ind][0] = self.class_map[r[0]]  # class
            gt[ind][1] = float(r[1])  # truncated
            gt[ind][2] = float(r[2])  # occluded
            gt[ind][3] = float(r[3])  # alpha
            gt[ind][4] = float(r[8])  # h
            gt[ind][5] = float(r[9])  # w
            gt[ind][6] = float(r[10])  # l
            gt[ind][7:10] = [float(r[11]), float(r[12]), float(r[13])]  # c
            gt[ind][10] = float(r[14])  # ry
            gt[ind][11] = item

        t = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        v = t[:, 0:3]
        # p = o3d.geometry.PointCloud()
        # p.points = o3d.utility.Vector3dVector(v)

        # p = p.voxel_down_sample(voxel_size=0.1)

        # t = np.asarray(p.points)

        t, _ = self.object_points(v, gt[0])
        t = v[t, :]

        if t.shape[0] < 200:
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        random_indices = np.random.choice(t.shape[0], self.npoints, replace=True)

        t = t[random_indices, :3]

        corresponding_bbox = np.ones((t.shape[0],)) * -1
        for i in inds:
            inside_points, _ = self.object_points(t, gt[i])
            corresponding_bbox[inside_points] = i

        t = t.astype(float)

        return t, gt, corresponding_bbox

    def dist(self, center: torch.Tensor, size: torch.Tensor, car_prob, gtt: torch.Tensor,
             bboxes: torch.Tensor, angle: torch.Tensor, seed_inds: torch.Tensor):

        center = center
        bboxes = bboxes.long()

        idx = int(gtt[0][11])
        R0, Tr = self.all_calib['R0'][idx], self.all_calib['Tr'][idx]

        R0 = torch.tensor(R0).double().to(device)
        Tr = torch.tensor(Tr).double().to(device)

        num = 0
        center_loss = torch.zeros(1).to(device)
        angle_loss = torch.zeros(1).to(device)
        size_loss = torch.zeros(1).to(device)

        for proposal in range(center.shape[0]):
            if bboxes[seed_inds[proposal]] == -1:
                continue

            num = num + 1

            gt = gtt[bboxes[seed_inds[proposal]]]

            l = size[proposal][2]
            w = size[proposal][1]
            h = size[proposal][0]

            corners_3d = torch.zeros(3, 9).to(device)  # the last column contains center coordinates
            zero_tensor = torch.tensor(0,dtype=torch.float).to(device)
            corners_3d[0, :] = torch.stack(
                [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, zero_tensor]).to(device)\
                               + center[proposal][0]
            corners_3d[1, :] = torch.stack([zero_tensor, zero_tensor, zero_tensor, zero_tensor, -h, -h, -h, -h, zero_tensor]).\
                                   to(device) + center[proposal][1]
            corners_3d[2, :] = torch.stack(
                [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2,zero_tensor]).to(device) + center[proposal][2]

            edges = torch.ones(4, 9).double().to(device)
            edges[0:3, 0:9] = corners_3d

            edges = R0 @ Tr @ edges

            edges = edges / edges[3, :]
            edges = edges[0:3, :]

            pred_size = self.find_bbox_size(edges.transpose(0, 1)).to(device)

            pred_center = edges[0:3, 8].to(device)

            gt_extents = gt[4:7].to(device)
            gt_center = gt[7:10].to(device)

            gt_angle = gt[10].to(device)

            center_loss = center_loss + torch.sum((gt_center - pred_center) ** 2)
            size_loss = size_loss + torch.sum((gt_extents - pred_size) ** 2)
            angle_loss = angle_loss + (gt_angle - angle[proposal]) ** 2

        return center_loss / num, size_loss / num, angle_loss / num

    def find_bbox_size(self, points):
        """
        :param points:
             points = tensor[8,3]
        :return:
            return tensor size[h,w,l] -->
        """
        l_max = torch.max(points[:, 0])
        l_min = torch.min(points[:, 0])

        h_max = torch.max(points[:, 1])
        h_min = torch.min(points[:, 1])

        w_max = torch.max(points[:, 2])
        w_min = torch.min(points[:, 2])

        return torch.stack([h_max - h_min, w_max - w_min, l_max - l_min])

    def object_points(self, p, gt):
        idx = int(gt[11])
        s = int(self.image_idx_list[idx])
        image = cv2.imread(os.path.join(self.image_dir, '%06d.png' % s))

        R = np.zeros((3, 3))
        R[0] = [np.cos(gt[10]), 0.0, np.sin(gt[10])]
        R[1] = [0, 1, 0]
        R[2] = [-np.sin(gt[10]), 0.0, np.cos(gt[10])]

        center = np.copy(gt[7:10])

        extents = np.ndarray((3,))
        extents[0] = gt[6]
        extents[1] = gt[5]
        extents[2] = gt[4]

        center[1] = center[1] - extents[1] / 2

        P2, R0, Tr = self.all_calib['P2'][idx], self.all_calib['R0'][idx], self.all_calib['Tr'][idx]

        center = center[0:3]
        extents = extents[0:3]

        y = np.ones((p.shape[0], 4))
        y[:, 0:3] = p
        y = np.transpose(y)

        y = R0 @ Tr @ y
        y = y / y[3, :]
        y = np.transpose(y)

        y = o3d.utility.Vector3dVector(y[:, 0:3])

        bb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extents)

        inv = np.linalg.pinv(Tr) @ np.linalg.pinv(R0)
        f = np.ndarray((4,))
        f[0:3] = center
        f[3] = 1
        c = inv @ f
        c = c / c[3]

        return bb.get_point_indices_within_bounding_box(y), c[0:3]

        # for index in bb.get_point_indices_within_bounding_box(y):
        #     s = abs(y[index] - y[bb.get_point_indices_within_bounding_box(y)[0]])
        #     print((s[0] + s[1] + s[2]) / 3)
        #     x = np.ones((4,))
        #     x[0:3] = y[index]
        #     z = P2 @ x
        #     z = z / z[2]
        #     cv2.circle(image, center=(int(z[0]), int(z[1])), radius=5, color=[0, 255, 255])
        #
        # cv2.imwrite("object_points" + str(idx) + ".png", image)

        # return bb.get_point_indices_within_bounding_box(y), c[0:3]


if __name__ == '__main__':
    train_set = KittyDataset('F:\data_object_velodyne', split='test')
    idx = 1000

    points, gt, corresponding_bbox = train_set.__getitem__(idx)

    rect_points = train_set.get_rect_points(gt[0])

    idx = int(gt[0][11])
    s = int(train_set.image_idx_list[idx])
    image = cv2.imread(os.path.join(train_set.image_dir, '%06d.png' % s))

    P2, R0, Tr = train_set.all_calib['P2'][idx], train_set.all_calib['R0'][idx], train_set.all_calib['Tr'][idx]
    lidar_points = np.linalg.pinv(Tr) @ np.linalg.pinv(R0) @ rect_points

    lidar_points = lidar_points / lidar_points[3, :]

    size = torch.zeros(1, 3).to(device)
    size[0] = train_set.find_bbox_size(torch.from_numpy(np.transpose(lidar_points[0:3, 0:8])).to(device))

    center1 = lidar_points[0, 8]
    center2 = lidar_points[1, 8]
    center3 = lidar_points[2, 8]

    center = torch.tensor([[center1, center2, center3]]).to(device)

    bboxes = torch.zeros(1).to(device)
    seed_inds = torch.zeros(1).to(torch.long).to(device)
    print(train_set.dist(center, size, 1, torch.from_numpy(gt).to(device), bboxes,
                         torch.from_numpy(gt[0][10:11]).to(device), seed_inds))
