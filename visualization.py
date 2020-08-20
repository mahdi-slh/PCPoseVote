import open3d as o3d
from kitty import KittyDataset
import kitty
from config import *
import os
from Network import Votenet

test_set = KittyDataset(PATH, split='train')


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


def get_gt_boundingbox(gt):
    R = np.zeros((3, 3))
    R[0] = [np.cos(gt[10]), 0.0, np.sin(gt[10])]
    R[1] = [0, 1, 0]
    R[2] = [-np.sin(gt[10]), 0.0, np.cos(gt[10])]
    center = gt[7:10].copy()
    center[1] = center[1] - gt[5] / 2

    bb = o3d.geometry.OrientedBoundingBox(center=center, R=R, extent=gt[6:3:-1])
    bb.color = [255, 255, 255]
    return bb


def get_pred_boundingbox(result, idx, angle,proposal_num):
    R0, Tr = test_set.all_calib['R0'][idx], test_set.all_calib['Tr'][idx]

    result[:, 1:7, :] = result[:, 1:7, :] * m[:, None, None]
    result[:, 1:4, :] = result[:, 1:4, :] + centroid[:, :, None]

    result = result[0,:,proposal_num].cpu().detach().numpy()

    edges = np.ones((4, 1))
    edges[0:3, 0] = result[1:4]

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
    bb.color = [255, 0, 0]
    return bb


vis = o3d.visualization.VisualizerWithKeyCallback()
vis.create_window(width=1920, height=1001, left=0, top=0)
vis.get_render_option().load_from_json('o3d_config/render_option.json')
param = o3d.io.read_pinhole_camera_parameters('o3d_config/view.json')

idx = 1223

data, gt, corresponding_bbox, centroid, m = test_set.__getitem__(idx)
x = data * m
x = x + centroid
p = get_points(x, idx)
p.colors = o3d.utility.Vector3dVector(np.ones(x.shape))
vis.add_geometry(p)

p = get_points(get_all_points(idx)[:, 0:3], idx)
vis.add_geometry(p)
vis.add_geometry(get_gt_boundingbox(gt[0]))

data = torch.from_numpy(data).to(device)
centroid = torch.from_numpy(centroid).to(device)[None, :]
m = torch.from_numpy(np.atleast_1d(m)).to(device)[None, :]
data = data.to(torch.float32)[None, :]

initial_inds = torch.unsqueeze(torch.arange(start=0, end=data.shape[1]), 0).repeat(data.shape[0], 1)

checkpoint = torch.load(SAVE_PATH)
model = Votenet(num_class=2, num_heading_bin=2, num_size_cluster=2, mean_size_arr=np.zeros((3, 1)),
                input_feature_dim=2)
model.to(device)
model.load_state_dict(checkpoint['model_state_dict'])

_, _, result, _ = model(data, initial_inds)

vis.add_geometry(get_pred_boundingbox(result, idx, gt[0][10],0))

vis.get_view_control().convert_from_pinhole_camera_parameters(param)
vis.run()
# vis.add(get_points(541))
