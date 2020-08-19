import torch
from kitty import KittyDataset
import numpy as np
from config import *


def box_dist(dataset, center: torch.Tensor, size: torch.Tensor, car_prob, gtt: torch.Tensor,
             bboxes: torch.Tensor, angle: torch.Tensor, seed_inds: torch.Tensor):
    bboxes = bboxes.long()
    idx = int(gtt[0][11])
    R0, Tr = dataset.all_calib['R0'][idx], dataset.all_calib['Tr'][idx]

    R0 = torch.tensor(R0).double().to(device)
    Tr = torch.tensor(Tr).double().to(device)

    num = 0
    center_loss = torch.zeros(1).to(device)
    size_loss = torch.zeros(1).to(device)
    car_prob_loss = torch.zeros(1).to(device)
    for proposal in range(center.shape[0]):
        if bboxes[seed_inds[proposal]] == -1:
            car_prob_loss = car_prob_loss - torch.log(1 - car_prob[proposal])
            # print("new loss -1: ", - torch.log(1 - car_prob[proposal]))
            continue

        num = num + 1

        gt = gtt[bboxes[seed_inds[proposal]]]

        edges = torch.ones(4, 1).double().to(device)
        edges[0:3, 0] = center[proposal]

        edges = R0 @ Tr @ edges

        edges = edges / edges[3, :]
        pred_center = edges[0:3,0]

        gt_extents = gt[4:7].to(device) - size_template
        gt_center = gt[7:10].to(device).clone()
        gt_center[1] = gt_center[1] - gt[4] / 2

        gt_angle = gt[10].to(device)

        center_loss = center_loss + (torch.sum((gt_center - pred_center) ** 2))
        size_loss = size_loss + (torch.sum((gt_extents - size[proposal]) ** 2))
        car_prob_loss = car_prob_loss - torch.log(car_prob[proposal])
        # print("new loss 1: ", - torch.log(car_prob[proposal]))

    if num == 0:
        num = 1
    return center_loss / num, size_loss / num, car_prob_loss / center.shape[0]


def compute_box_loss(gt: torch.Tensor, corresponding_bbox: torch.Tensor, output: torch.Tensor, seed_inds: torch.Tensor,
                     train_dataset: KittyDataset):
    B, _, M = output.shape

    total_center_loss = torch.zeros(B, ).to(device)
    total_size_loss = torch.zeros(B, ).to(device)
    total_car_prob_loss = torch.zeros(B, ).to(device)

    for i in range(B):
        car_prob = 1 / (1 + torch.exp(-output[i, 0, :]))
        center = output[i, 1:4, :].transpose(1, 0)
        size = output[i, 4:7, :].transpose(1, 0)
        angle = output[i, 7]

        center_loss, size_loss, car_prob_loss = box_dist(train_dataset, center, size, car_prob, gt[i],
                                                         corresponding_bbox[i], angle, seed_inds[i])

        total_center_loss[i] = center_loss
        total_size_loss[i] = size_loss
        total_car_prob_loss[i] = car_prob_loss

    return torch.mean(total_center_loss), torch.mean(total_size_loss), torch.mean(total_car_prob_loss)


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
            if len(list) != 0:
                loss[i] = loss[i] + 3 * torch.mean(torch.abs(vote_xyz[i, list] - torch.from_numpy(gt_vote).to(device)))

    return torch.mean(loss)
