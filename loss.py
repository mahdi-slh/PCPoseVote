import torch
from kitty import KittyDataset
import numpy as np
from config import *


def compute_box_loss(gt: torch.Tensor, corresponding_bbox: torch.Tensor, output: torch.Tensor, seed_inds: torch.Tensor,
                     train_dataset: KittyDataset):
    B, _, M = output.shape

    total_center_loss = torch.zeros(B, ).to(device)
    total_size_loss = torch.zeros(B, ).to(device)
    total_angle_loss = torch.zeros(B, ).to(device)

    for i in range(B):
        car_prob = 1 / (1 + torch.exp(output[i, 0, :]))
        center = output[i, 1:4, :].transpose(1, 0)
        size = output[i, 4:7, :].transpose(1, 0)
        angle = output[i, 7]

        center_loss, size_loss, angle_loss = train_dataset.dist(center, size, car_prob, gt[i], corresponding_bbox[i],
                                                                angle, seed_inds[i])

        total_center_loss[i] = total_center_loss[i] + center_loss
        total_size_loss[i] = total_size_loss[i] + size_loss
        total_angle_loss[i] = total_angle_loss[i] + angle_loss

    return torch.mean(total_center_loss), torch.mean(total_size_loss) , torch.mean(total_angle_loss)


def compute_vote_loss(gt: torch.Tensor, xyz: torch.Tensor, vote_xyz: torch.Tensor, train_dataset: KittyDataset):
    """
        xyz.shape = [B,3,M]
        vote_xyz.shape = [B,3,M]
    """

    xyz = xyz.permute(0, 2, 1)

    B, M, _ = xyz.shape

    loss = torch.zeros(B, device=xyz.device)
    xyz_numpy = xyz.cpu().detach().numpy()
    gt_numpy = gt.cpu().detach().numpy()
    for i in range(B):
        for k in range(len(gt[i])):
            if gt[i][k][0] == -1:
                continue

            list, gt_vote = train_dataset.object_points(xyz_numpy[i], gt_numpy[i][k])
            m_pos = len(list)
            if m_pos != 0:
                loss[i] = loss[i].add(
                    torch.sum(torch.abs(vote_xyz[i, list] - torch.from_numpy(gt_vote).to(device)), [0, 1]) / m_pos).to(
                    device)

    return torch.mean(loss, -1)
