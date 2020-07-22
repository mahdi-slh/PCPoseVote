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

        corners_3d = np.ndarray((3, 8))
        corners_3d[0] = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
        corners_3d[1] = [0, 0, 0, 0, -h, -h, -h, -h]
        corners_3d[2] = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

        corners_3d = np.matmul(R, corners_3d)

        corners_3d[0, :] = corners_3d[0, :] + gt[7]
        corners_3d[1, :] = corners_3d[1, :] + gt[8]
        corners_3d[2, :] = corners_3d[2, :] + gt[9]

        points_3d = np.ndarray((4, 8))
        points_3d[0:3, :] = corners_3d
        points_3d[3, :] = [1] * 8

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

    def show_points(self, points, gt):
        idx = int(gt[11])
        s = int(self.image_idx_list[idx])
        image = cv2.imread(os.path.join(self.image_dir, '%06d.png' % s))

        P2, R0, Tr = train_set.all_calib['P2'][idx], train_set.all_calib['R0'][idx], train_set.all_calib['Tr'][idx]

        x = np.ones((points.shape[0], 4))
        x[0:points.shape[0], 0:3] = points
        y = np.transpose(x)
        y = R0 @ Tr @ y
        y = y / y[3, :]

        z = self.get_rect_points(gt)

        l = abs(z[0][0] - z[0][7])
        h = abs(z[1][0] - z[1][7])
        w = abs(z[2][0] - z[2][1])

        center1 = (z[0][0] + z[0][7]) / 2
        center2 = (z[1][0] + z[1][7]) / 2
        center3 = (z[2][0] + z[2][1]) / 2

        # all_points = np.ndarray((3, 0))

        r = []
        for i in range(points.shape[0]):
            on_surface1 = abs(y[0][i] - center1) < l / 2
            on_surface2 = abs(center2 - y[1][i]) < h / 2
            on_surface3 = abs(y[2][i] - center3) < w / 2

            on_surface1 = True
            on_surface2 = True
            on_surface3 = True

            # flag = self.is_equal(points[i][0], points[i][1], points[i][2], 1, 46.78826479, -2.84346198, -2.13290362, 1)

            if on_surface1 and on_surface2 and on_surface3:
                r.append(i)

        all_points = np.transpose(points[r])
        points1 = np.ones((4, all_points.shape[1]))
        points1[0:3, :] = all_points
        all_points = P2 @ R0 @ Tr @ points1

        for i in range(all_points.shape[1]):
            cv2.circle(image,
                       center=(int(all_points[0][i] / all_points[2][i]), int(all_points[1][i] / all_points[2][i])),
                       radius=1, color=[0, 255, 255])
        self.temp = self.temp + 1
        cv2.imwrite("segmentation" + str(self.temp) + ".png", image)

        inv = np.linalg.pinv(Tr) @ np.linalg.pinv(R0)
        f = np.ndarray((4, 1))
        f[0, 0] = center1
        f[1, 0] = center2
        f[2, 0] = center3
        f[3, 0] = 1
        c = inv @ f
        c = c / c[3]
        return r, c[0:3, 0]

    def draw_3dBox(self, gt):
        idx = int(gt[11])
        s = int(self.image_idx_list[idx])
        image = cv2.imread(os.path.join(self.image_dir, '%06d.png' % s))

        points_3d = self.get_rect_points(gt)

        P2, _, _ = train_set.all_calib['P2'][idx], train_set.all_calib['R0'][idx], train_set.all_calib['Tr'][idx]

        pts_2D = np.matmul(P2, points_3d)
        pts_2D[0, :] = pts_2D[0, :] / pts_2D[2, :]
        pts_2D[1, :] = pts_2D[1, :] / pts_2D[2, :]

        pts_2D = pts_2D[0:2, :]

        for i in range(0, 7):
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
        for i in range(len(lines)):
            r = lines[i].split(' ')
            if self.class_map[r[0]] not in self.classes:
                continue
            ind = ind + 1
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
        t = t.astype(float)

        return t, gt

    def dist(self, center: torch.Tensor, size: torch.Tensor, car_prob, gtt: torch.Tensor):

        idx = int(gtt[0][11])
        P2, R0, Tr = self.all_calib['P2'][idx], self.all_calib['R0'][idx], self.all_calib['Tr'][idx]

        R0 = torch.tensor(R0).double().to(device)
        Tr = torch.tensor(Tr).double().to(device)

        y_center = torch.ones(center.shape[0], 4).double().to(device)
        y_center[:, 0:3] = center.clone().to(device)
        y_center = y_center.transpose(0, 1)
        y_center = R0 @ Tr @ y_center
        y_center = y_center / y_center[3, :]
        y_center = y_center.transpose(0, 1)
        y_center = y_center[:, 0:3]

        y_size = torch.ones(center.shape[0], 4).double().to(device)
        y_size[:, 0:3] = size.clone().to(device)
        y_size = y_size.transpose(0, 1)
        y_size = R0 @ Tr @ y_size
        y_size = y_size / y_size[3, :]
        y_size = y_size.transpose(0, 1)
        y_size = y_size[:, 0:3]

        dist = torch.ones(y_center.shape[0]).to(device) * 1
        indices = np.ones((y_center.shape[0],)) * -1
        for j in range(gtt.shape[0]):
            gt = gtt[j]

            R = torch.zeros(3, 3)
            R[0] = torch.tensor([torch.cos(gt[10]), 0.0, torch.sin(gt[10])])
            R[1] = torch.tensor([0, 1, 0])
            R[2] = torch.tensor([-torch.sin(gt[10]), 0.0, torch.cos(gt[10])])
            center = (gt[7:10]).detach().clone().to(device)
            extents = torch.zeros(3)
            extents[0] = (gt[6]).detach().clone()
            extents[1] = (gt[5]).detach().clone()
            extents[2] = (gt[4]).detach().clone()
            center[1] = center[1] - extents[1] / 2
            for i in range(y_center.shape[0]):
                if dist[i] < torch.mean(torch.abs(y_center[i] - center)).to(device):
                    indices[i] = j

        loss = torch.zeros(1).to(device)
        for i in range(y_center.shape[0]):
            if indices[i] != -1:
                gt = gtt[int(indices[i])]
                R = torch.zeros(3, 3)
                R[0] = torch.tensor([torch.cos(gt[10]), 0.0, torch.sin(gt[10])])
                R[1] = torch.tensor([0, 1, 0])
                R[2] = torch.tensor([-torch.sin(gt[10]), 0.0, torch.cos(gt[10])])
                center = (gt[7:10]).detach().clone().to(device)
                extents = torch.zeros(3).to(device)
                extents[0] = (gt[6]).detach().clone().to(device)
                extents[1] = (gt[5]).detach().clone().to(device)
                extents[2] = (gt[4]).detach().clone().to(device)
                center[1] = center[1] - extents[1] / 2
                loss = loss + torch.mean(torch.abs(y_center[i] - center) + torch.abs(y_size[i] - extents)) - torch.log(
                    car_prob[i])
            else:
                loss = loss - torch.log(1 - car_prob[i])

        return loss

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
    idx = 1238

    points, gt = train_set.__getitem__(idx)
    # train_set.draw_3dBox(gt[1])
    # train_set.show_points(points, gt[0])
    train_set.object_points(points, gt[0])
