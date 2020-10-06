import os
import torch.utils.data as torch_data
import numpy as np
import cv2
import torch
import open3d as o3d
import random
from config import *
from pointcloud_partitioning import pc_partition
from pointcloud_partitioning import farthest_point_sample

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return centroid, m, pc


class KittyDataset(torch_data.Dataset):
    def __init__(self, root_dir, npoints=data_points, split='train', mode='TRAIN', random_select=True):
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

    def get_box_points(self, gt, from_gt=True):

        R = np.ndarray((3, 3))
        R[0] = [np.cos(gt[10]), 0, np.sin(gt[10])]
        R[1] = [0, 1, 0]
        R[2] = [-np.sin(gt[10]), 0, np.cos(gt[10])]

        l = np.copy(gt[6])
        w = np.copy(gt[5])
        h = np.copy(gt[4])

        corners_3d = np.ndarray((3, 9))
        corners_3d[0] = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2, 0]
        corners_3d[1] = [h / 2, h / 2, h / 2, h / 2, -h / 2, -h / 2, -h / 2, -h / 2, 0]
        corners_3d[2] = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2, 0]

        corners_3d = R @ corners_3d

        corners_3d[0, :] = corners_3d[0, :] + gt[7]
        corners_3d[1, :] = corners_3d[1, :] + gt[8]
        if from_gt:
            corners_3d[1, :] = corners_3d[1, :] - h / 2
        corners_3d[2, :] = corners_3d[2, :] + gt[9]

        points_3d = np.ndarray((4, 9))
        points_3d[0:3, :] = corners_3d
        points_3d[3, :] = [1] * 9

        return points_3d

    def __len__(self):
        return len(self.image_idx_list)

    def __getitem__(self, item):
        R0, Tr = self.all_calib['R0'][item], self.all_calib['Tr'][item]

        s = int(self.image_idx_list[item])
        path = os.path.join(self.lidar_dir, '%06d.bin' % s)
        label_path = os.path.join(self.label_dir, '%06d.txt' % s)
        t = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
        t = t.astype('float32')
        v = t[:, 0:3]

        x = np.bitwise_and(v[:, 0] < 100, v[:, 0] > 0)
        x = np.bitwise_and(x, v[:, 2] < 10)
        x = np.bitwise_and(x, v[:, 2] > -10)
        v = v[x]


        lines = [x.strip() for x in open(label_path).readlines()]
        gt = -np.ones((self.max_objects, 15)) * 1000
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
            gt[ind][7:10] = (np.array([float(r[11]), float(r[12]), float(r[13])]))  # c
            gt[ind][10] = float(r[14])  # ry
            gt[ind][11] = item

        # p = o3d.geometry.PointCloud()
        # p.points = o3d.utility.Vector3dVector(v)

        # p = p.voxel_down_sample(voxel_size=0.1)

        # t = np.asarray(p.points)
        contain_car = np.random.choice([False, True], size=(1,), p=[0.3, 0.7])
        if ind == -1:
            return self.__getitem__(np.random.randint(low=0, high=self.__len__()))
        rnd = np.random.randint(low=0, high=ind + 1)
        t, box_data = self.object_points(v, gt[rnd], random=True,contain_car=contain_car[0])
        t = v[t, :]

        if t.shape[0] < minimum_number_of_points:
            return self.__getitem__(np.random.randint(low=0, high=self.__len__()))

        # x = np.bitwise_and(v[:, 0] < 100, v[:, 0] > 0)
        # x = np.bitwise_and(x, v[:, 2] < 10)
        # x = np.bitwise_and(x, v[:, 2] > -10)
        # v = v[x]

        # _,regions = pc_partition(torch.from_numpy(v[np.newaxis,:,:]).to('cpu'))
        # regions = regions[0].cpu().detach().numpy()
        # random_slice = np.random.choice(farthest_point_num,(farthest_point_num,),replace=False)

        # for i in range(random_slice.shape[0]):
        #     if np.unique(regions[random_slice[i]]).shape[0] < 400:
        #         continue
        #     t = v[regions[random_slice[i]], :]
        #     break

        random_indices = np.random.choice(t.shape[0], self.npoints, replace=True)
        t = t[random_indices, :3]

        corresponding_bbox = np.ones((t.shape[0],)) * -1000
        for i in inds:
            inside_points, _ = self.object_points(t, gt[i])
            corresponding_bbox[inside_points] = i
        gt[:ind + 1, 12:15] = gt[:ind + 1, 7:10]
        gt[:ind + 1, 13] = gt[:ind + 1, 13] - gt[:ind + 1, 5] / 2

        inv = np.linalg.pinv(Tr) @ np.linalg.pinv(R0)
        f = np.ones((4, ind + 1))
        f[0:3, :] = np.transpose(gt[:ind+1, 12:15])
        c = inv @ f
        c = c / c[3]
        gt[:ind+1, 12:15] = np.transpose(c[0:3, :])

        centroid, m, t = pc_normalize(t)
        t = t.astype(float)

        return t, gt, corresponding_bbox, centroid, m, rnd, box_data

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

    def object_points(self, p, gt, random=False, contain_car = True):
        idx = int(gt[11])
        P2, R0, Tr = self.all_calib['P2'][idx], self.all_calib['R0'][idx], self.all_calib['Tr'][idx]

        # s = int(self.image_idx_list[idx])
        # image = cv2.imread(os.path.join(self.image_dir, '%06d.png' % s))

        R = np.zeros((3, 3))
        R[0] = [np.cos(gt[10]), 0.0, np.sin(gt[10])]
        R[1] = [0, 1, 0]
        R[2] = [-np.sin(gt[10]), 0.0, np.cos(gt[10])]

        box_data = np.zeros(6,) # 0:3 --> center , 3:6 --> width
        if contain_car:
            center = np.copy(gt[7:10])
            extents = np.ndarray((3,))
            extents[0] = gt[6]
            extents[1] = gt[5]
            extents[2] = gt[4]
            center[1] = center[1] - extents[1] / 2
            box_data[3:6] = np.copy(extents)
        else:
            # center = np.random.normal([-5,0,15],2,(3,))
            # center[1] = 0.7

            p_prime = torch.from_numpy(p[np.newaxis,:,:])
            fps = farthest_point_sample(p_prime, farthest_point_num)
            fps_points = torch.gather(p_prime, 1, fps[0, :, None].repeat(1, 1, 3))[0]

            y = o3d.utility.Vector3dVector(p)
            for i in np.random.permutation(farthest_point_num):
                extents = np.flipud(sample_box_size)
                bb = o3d.geometry.OrientedBoundingBox(center=fps_points[i].numpy(), R=np.eye(3),
                                                      extent=extents)
                if len(bb.get_point_indices_within_bounding_box(y)) >= minimum_number_of_points:
                    box_data[0:3] = np.copy(fps_points[i])
                    extents = np.flipud(sample_box_size)
                    box_data[3:6] = np.copy(extents)
                    return bb.get_point_indices_within_bounding_box(y), box_data



        box_data[0:3] = np.copy(center)


        y = np.ones((p.shape[0], 4))
        y[:, 0:3] = p
        y = np.transpose(y)

        y = R0 @ Tr @ y
        y = y / y[3, :]
        y = np.transpose(y)

        y = o3d.utility.Vector3dVector(y[:, 0:3])

        if not random:
            bb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=extents)
        else:
            center = center + np.random.rand(3)*random_shift_interval
            box_data[0:3] = np.copy(center)
            extents = np.flipud(sample_box_size)
            box_data[3:6] = np.copy(extents)
            bb = o3d.geometry.OrientedBoundingBox(center=center , R=np.eye(3),
                                                  extent=extents)


        return bb.get_point_indices_within_bounding_box(y),box_data

        # for index in bb.get_point_indices_within_bounding_box(y):
        #     s = abs(y[index] - y[bb.get_point_indices_within_bounding_box(y)[0]])
        #     # print((s[0] + s[1] + s[2]) / 3)
        #     x = np.ones((4,))
        #     x[0:3] = y[index]
        #     z = P2 @ x
        #     z = z / z[2]
        #     cv2.circle(image, center=(int(z[0]), int(z[1])), radius=5, color=[0, 255, 255])
        #
        # cv2.imwrite("object_points" + str(idx) + ".png", image)
        #
        # return bb.get_point_indices_within_bounding_box(y), c[0:3]


# for debugging purpose
if __name__ == '__main__':
    train_set = KittyDataset('F:\data_object_velodyne', split='test')
    idx = 641
    points, gt, corresponding_bbox, center, m = train_set.__getitem__(idx)
    print(train_set.find_bbox_size(points))
    exit(1)

    train_set.object_points(points, gt[0])
    exit(1)

    rect_points = train_set.get_box_points(gt[0])

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

    # print(loss.box_dist(train_set,center, size, 1, torch.from_numpy(gt).to(device), bboxes,
    #                          torch.from_numpy(gt[0][10:11]).to(device), seed_inds))
