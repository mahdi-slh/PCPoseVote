import torch
from kitty import KittyDataset
import numpy as np
from config import *


def compute_box_loss(gt: torch.Tensor, objectness_label: torch.Tensor, objectness_mask: torch.Tensor,
                     objectness_assignment: torch.Tensor, result: torch.Tensor):

    pred_size = result[:, :, 4:7] + size_template
    pred_center = result[:, :, 1:4]

    pred_car_prob = 1 / (1 + torch.exp(-result[:, :, 0]))
    car_prob = torch.zeros(pred_car_prob.shape[0], pred_car_prob.shape[1], 2).to(device)
    car_prob[:, :, 0] = 1-pred_car_prob
    car_prob[:, :, 1] = pred_car_prob

    x = objectness_assignment[:, :, None].repeat(1, 1, 3)
    gt_center = torch.gather(gt[:, :, 12:15], 1, x)
    gt_size = torch.gather(gt[:, :, 4:7], 1, x)

    total = (torch.sum(objectness_label, 1))+ 1e-6
    center_loss = torch.sum(torch.sum((pred_center - gt_center) ** 2, 2) * objectness_label,1) / total
    size_loss = torch.sum(torch.sum((pred_size - gt_size) ** 2, 2) * objectness_label,1) / total
    criterion = torch.nn.CrossEntropyLoss(torch.from_numpy(OBJECTNESS_CLS_WEIGHTS).to(device).to(torch.float32),
                                          reduction='none')

    target_labels = (objectness_label > 0).to(torch.long)


    car_prob_loss = criterion(car_prob.transpose(1,2), target_labels)
    car_prob_loss = torch.sum(car_prob_loss * objectness_mask)/(torch.sum(objectness_mask)+1e-6)

    return torch.mean(center_loss), torch.mean(size_loss), car_prob_loss
