import open3d as o3d
from kitty import KittyDataset
import kitty
import torch.utils.data as torch_data
from config import *
import os
from Network import Votenet
from nn_distance import nn_distance
from pointcloud_partitioning import pc_partition
import numpy as np

test_set = KittyDataset(PATH, split='test')
test_loader = torch_data.DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=1)


def save_view_point(vis):
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters("o3d_config/view.json", param)
    print("View point saved")


def get_points(points, idx):
    v = np.ones((points.shape[0], 4))
    v[:, 0:3] = points[:, 0:3]
    R0, Tr = test_set.all_calib['R0'][idx], test_set.all_calib['Tr'][idx]
    v = R0 @ Tr @ np.transpose(v)
    v = v / v[3]
    v = np.transpose(v[0:3, :])

    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(v)
    return p


def get_all_points(idx):
    s = int(test_set.image_idx_list[idx])
    path = os.path.join(test_set.lidar_dir, '%06d.bin' % s)
    data = np.fromfile(path, dtype=np.float32).reshape(-1, 4)
    data = data.astype('float32')
    return data


def add_gt_boundingbox(gtt, box_data):
    for i in range(gtt.shape[0]):
        gt = gtt[i]
        if gt[0] == -1000:
            break
        R = np.zeros((3, 3))
        R[0] = [np.cos(gt[10]), 0.0, np.sin(gt[10])]
        R[1] = [0, 1, 0]
        R[2] = [-np.sin(gt[10]), 0.0, np.cos(gt[10])]
        center = gt[7:10].copy()
        center[1] = center[1] - gt[5] / 2
        bb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=gt[6:3:-1])
        bb.color = [1, 1, 1]
        vis.add_geometry(bb)

    bb2 = o3d.geometry.OrientedBoundingBox(center=box_data[0:3], R=np.eye(3), extent=box_data[3:6])
    bb2.color = [0.82, 0.41, 0.12]
    vis.add_geometry(bb2)


def add_all_gt_boundingboxes(gtt):
    for i in range(gtt.shape[0]):
        gt = gtt[i]
        if gt[0] == -1000:
            break
        R = np.zeros((3, 3))
        R[0] = [np.cos(gt[10]), 0.0, np.sin(gt[10])]
        R[1] = [0, 1, 0]
        R[2] = [-np.sin(gt[10]), 0.0, np.cos(gt[10])]
        center = gt[7:10].copy()
        center[1] = center[1] - gt[5] / 2
        bb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=gt[6:3:-1])
        bb.color = [1, 1, 1]
        vis.add_geometry(bb)


# def add_pred_boundingbox(all_results, idx, angles, all_proposals, ind1, batch_index):
#     R0, Tr = test_set.all_calib['R0'][idx], test_set.all_calib['Tr'][idx]
#     all_results[1:7, :] = all_results[1:7, :] * m[batch_index, None, None]
#     all_results[1:4, :] = all_results[1:4, :] + centroid[batch_index, :, None]
#
#     for proposal_num in all_proposals:
#         angle = angles[ind1[batch_index, proposal_num]]
#         result = all_results[:, proposal_num].cpu().detach().numpy()
#
#         edges = np.ones((4, 1))
#         edges[0:3, 0] = result[1:4].copy()
#
#         edges = R0 @ Tr @ edges
#
#         edges = edges / edges[3, :]
#         pred_center = edges[0:3, 0]
#         pred_size = result[4:7] + size_template.cpu().detach().numpy()
#         pred_size = np.flipud(pred_size)
#
#         R = np.zeros((3, 3))
#         R[0] = [np.cos(angle), 0.0, np.sin(angle)]
#         R[1] = [0, 1, 0]
#         R[2] = [-np.sin(angle), 0.0, np.cos(angle)]
#
#         bb = o3d.geometry.OrientedBoundingBox(center=pred_center, R=R, extent=pred_size)
#         bb.color = [1, 0, 0]
#         vis.add_geometry(bb)

def add_pred_bb(all_results, idx, angles, proposals, ind):
    R0, Tr = test_set.all_calib['R0'][idx], test_set.all_calib['Tr'][idx]
    angles = angles.detach().cpu().numpy()
    for proposal_num in proposals:
        angle = angles[ind[proposal_num]]
        result = all_results[:, proposal_num]
        edges = np.ones((4, 1))
        edges[0:3, 0] = result[1:4].copy()

        edges = R0 @ Tr @ edges

        edges = edges / edges[3, :]
        pred_center = edges[0:3, 0]
        pred_size = result[4:7] + size_template.cpu().detach().numpy()
        pred_size = np.flipud(pred_size)

        R = np.zeros((3, 3))
        R[0] = [np.cos(angle), 0.0, np.sin(angle)]
        R[1] = [0, 1, 0]
        R[2] = [-np.sin(angle), 0.0, np.cos(angle)]

        bb = o3d.geometry.OrientedBoundingBox(center=pred_center, R=R, extent=pred_size)
        bb.color = [1, 0, 0]
        vis.add_geometry(bb)


def get_gt(idx):
    R0, Tr = test_set.all_calib['R0'][idx], test_set.all_calib['Tr'][idx]

    s = int(test_set.image_idx_list[idx])
    label_path = os.path.join(test_set.label_dir, '%06d.txt' % s)

    lines = [x.strip() for x in open(label_path).readlines()]
    gt = -np.ones((test_set.max_objects, 15)) * 1000
    ind = -1
    inds = []
    for i in range(len(lines)):
        r = lines[i].split(' ')
        if test_set.class_map[r[0]] not in test_set.classes:
            continue
        ind = ind + 1
        inds.append(ind)
        gt[ind][0] = test_set.class_map[r[0]]  # class
        gt[ind][1] = float(r[1])  # truncated
        gt[ind][2] = float(r[2])  # occluded
        gt[ind][3] = float(r[3])  # alpha
        gt[ind][4] = float(r[8])  # h
        gt[ind][5] = float(r[9])  # w
        gt[ind][6] = float(r[10])  # l
        gt[ind][7:10] = (np.array([float(r[11]), float(r[12]), float(r[13])]))  # c
        gt[ind][10] = float(r[14])  # ry
        gt[ind][11] = idx
    gt[:ind + 1, 12:15] = gt[:ind + 1, 7:10]
    gt[:ind + 1, 13] = gt[:ind + 1, 13] - gt[:ind + 1, 5] / 2

    inv = np.linalg.pinv(Tr) @ np.linalg.pinv(R0)
    f = np.ones((4, ind + 1))
    f[0:3, :] = np.transpose(gt[:ind + 1, 12:15])
    c = inv @ f
    c = c / c[3]
    gt[:ind + 1, 12:15] = np.transpose(c[0:3, :])
    return gt


def interpret_result(result, gt):
    """
        :param result: [batch_size * num_proposal,8]
    """
    gt = torch.from_numpy(gt).to(device)
    result_rev = result.transpose(0, 1)
    pred_car_prob = torch.nn.functional.softmax(result[:,7:9])[:,1]
    carprob_argsort = torch.argsort(pred_car_prob, 0)
    print(pred_car_prob[carprob_argsort])
    _, ind, _, _ = nn_distance(result[None, :, 1:4], gt[None, :, 12:15])

    add_pred_bb(result_rev.detach().cpu().numpy(), idx, gt[:, 10],(pred_car_prob>0.75).nonzero(), ind[0])


if __name__ == '__main__':
    torch.manual_seed(414)
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window(width=1920, height=1001, left=0, top=30)
    vis.get_render_option().load_from_json('o3d_config/render_option.json')
    param = o3d.io.read_pinhole_camera_parameters('o3d_config/view.json')

    # data, gt, corresponding_bbox, centroid, m, rnd, box_data= next(iter(test_loader))
    # data = data.to(torch.float32)
    # data = data.to(device)
    # gt = gt.to(device)
    # corresponding_bbox = corresponding_bbox.to(device)
    # centroid = centroid.to(device)
    # m = m.to(device)
    #
    # batch_index = 5
    # idx = int(gt[batch_index, 0, 11].detach().cpu().numpy())

    idx = 988
    main_points = get_all_points(idx)[:, 0:3]
    x = np.bitwise_and(main_points[:, 0] < 100, main_points[:, 0] > 0)
    x = np.bitwise_and(x, main_points[:, 2] < 10)
    x = np.bitwise_and(x, main_points[:, 2] > -10)
    main_points = main_points[x]
    main_points_vis = get_points(main_points, idx)

    fps_points, partitions = pc_partition(torch.from_numpy(main_points[np.newaxis, :, :]).to(device))

    partitions = partitions.detach().cpu().numpy()
    fps_points = fps_points.detach().cpu().numpy()

    x = get_points(fps_points[0], idx)
    x = np.asarray(x.points)
    for i in range(0, x.shape[0]):
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.1)
        sphere = sphere.translate(x[i])
        vis.add_geometry(sphere)

    checkpoint = torch.load(SAVE_PATH)
    model = Votenet(num_class=2, num_heading_bin=2, num_size_cluster=2, mean_size_arr=np.zeros((3, 1)),
                    input_feature_dim=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    data = main_points[partitions[0]]
    input = torch.zeros(data.shape[0], test_points, 3).to(device)
    centroid = torch.zeros(data.shape[0], 3).to(device)
    m = torch.zeros(data.shape[0]).to(device)
    for i in range(64):
        c_, m_, pc = kitty.pc_normalize(data[i])
        input[i] = torch.from_numpy(pc).to(device)
        m[i] = torch.from_numpy(np.array([m_])).to(device)
        centroid[i] = torch.from_numpy(c_).to(device)

    all_results = torch.zeros(64,9,64).to(device)

    for i in range(4):
        with torch.no_grad():
            initial_inds = torch.unsqueeze(torch.arange(start=0, end=input.shape[1]), 0).repeat(batch_size, 1).to(
                device)
            _, _, result, _, _ = model(input[i*batch_size:(i+1)*batch_size], initial_inds)
            all_results[i*batch_size:(i+1)*batch_size] = result.clone()
            all_results[i*batch_size:(i+1)*batch_size,1:7, :] = all_results[i*batch_size:(i+1)*batch_size,1:7, :] * m[i*batch_size:(i+1)*batch_size, None, None]
            all_results[i*batch_size:(i+1)*batch_size:, 1:4, :] = all_results[i*batch_size:(i+1)*batch_size, 1:4, :] + centroid[i*batch_size:(i+1)*batch_size, :, None]
    all_results = all_results.transpose(1,2)

    all_results = all_results.reshape(-1, 9)
    # all_results = all_results[14]
    # for i in range(partitions.shape[1]):
    for i in range(0,64):
        x = main_points[partitions[0, i]]
        region_points = get_points(x, idx)
        colors = np.random.rand(3)[np.newaxis, :].repeat(x.shape[0], 0)
        region_points.colors = o3d.utility.Vector3dVector(colors)
        vis.add_geometry(region_points)

    colors = np.array([0.3, 0.3, 0.3])[np.newaxis, :].repeat(main_points.shape[0], 0)
    main_points_vis.colors = o3d.utility.Vector3dVector(colors)
    # vis.add_geometry(main_points_vis)

    # idx = int(gt[batch_index, 0, 11].detach().cpu().numpy())
    # with torch.no_grad():
    #     x = data[batch_index].cpu().numpy() * m.cpu().numpy()[batch_index]
    #     x = x + centroid[batch_index].cpu().numpy()
    #     p = get_points(x, idx)
    #     colors = np.array([1, 0.5, 0])[np.newaxis, :].repeat(x.shape[0], 0)
    #     p.colors = o3d.utility.Vector3dVector(colors)
    #     vis.add_geometry(p)

    # p = get_points(get_all_points(idx)[:, 0:3], idx)
    # vis.add_geometry(p)
    # print("gt object:", rnd[batch_index])
    gt = get_gt(idx)
    add_all_gt_boundingboxes(gt)
    interpret_result(all_results, gt)

    # initial_inds = torch.unsqueeze(torch.arange(start=0, end=data.shape[1]), 0).repeat(data.shape[0], 1)
    # aggregated_xyz, vote_xyz, result, seed_inds, vote_inds = model(data, initial_inds)
    # pred_car_prob = 1 / (1 + torch.exp(-result[:, 0, :]))
    # carprob_argsort = torch.argsort(pred_car_prob[batch_index, :], 0)
    # print(pred_car_prob[batch_index, carprob_argsort])
    # print(carprob_argsort)

    # with torch.no_grad():
    #     x = vote_xyz[batch_index].cpu().numpy() * m.cpu().numpy()[batch_index]
    #     x = x + centroid[batch_index].cpu().numpy()
    #     p = get_points(x, idx)
    #     colors = np.array([0.5, 0.2, 0.7])[np.newaxis, :].repeat(256, 0)
    #     p.colors = o3d.utility.Vector3dVector(colors)
    #     vis.add_geometry(p)

    # vis.add_geometry(get_points())

    # gt_object = torch.gather(corresponding_bbox, 1, vote_inds.to(torch.long)).to(torch.long).to(device)

    # vote_mask = (gt_object >= 0).byte().to(device)
    # gt_object[gt_object == -1] = 0

    # gtt = torch.from_numpy(gt).to(device)[None,:]
    # gt_vote = torch.gather(gtt[:, :, 12:15], 1, gt_object[:, :, None].repeat(1, 1, 3))
    # aggregated_xyz = aggregated_xyz.transpose(1, 2)
    # temp = aggregated_xyz * m[:, None, None]
    # temp = temp + centroid[:, None, :]

    # dist1, ind1, _, _ = nn_distance(temp, gt[:, :, 12:15])
    # s = torch.argsort(dist1, 1)
    # s = s[batch_index]
    # euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
    # last_positive = torch.nonzero(dist1[batch_index, s] < NEAR_THRESHOLD, as_tuple=False)[-1]
    # first_negative = torch.nonzero(dist1[batch_index, s] > FAR_THRESHOLD, as_tuple=False)[0]
    # print("last positive", last_positive)
    # print("first negative", first_negative)
    # print(last_positive)
    # print(dist1[batch_index, s])
    # print(ind1[batch_index, s])

    # add_pred_boundingbox(result[batch_index].clone(), idx, gt[batch_index, :, 10].detach().cpu().numpy(),
    #                      carprob_argsort[-5:-1], ind1,batch_index)

    vis.get_view_control().convert_from_pinhole_camera_parameters(param)
    vis.run()
