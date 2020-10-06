from Network import Votenet
import torch.utils.data as torch_data
from kitty import KittyDataset
import numpy as np
import cv2
import os
import loss
import torch
from config import *
from tqdm import tqdm
from matplotlib import pyplot as plt
import open3d as o3d
from nn_distance import nn_distance

torch.multiprocessing.freeze_support()

test_set = KittyDataset(PATH, split='train')
test_loader = torch_data.DataLoader(test_set, shuffle=True, batch_size=batch_size, num_workers=1)
model = Votenet(num_class=2, num_heading_bin=2, num_size_cluster=2, mean_size_arr=np.zeros((3, 1)),
                input_feature_dim=2)
torch.multiprocessing.freeze_support()

checkpoint = torch.load(SAVE_PATH)
model.load_state_dict(checkpoint['model_state_dict'])
epoch = checkpoint['epoch']
loss1 = checkpoint['loss']

if __name__ == '__main__':
    model.to(device)
    torch.multiprocessing.freeze_support()

    center_loss_values = []
    vote_loss_values = []
    size_loss_values = []
    total_loss_values = []
    car_prob_loss_values = []

    with torch.no_grad():
        for i, data in tqdm(enumerate(test_loader, 0)):
            print(i)
            data, gt, corresponding_bbox, centroid, m,_ = data
            # data, gt, corresponding_bbox = data

            data = data.to(torch.float32)
            data = data.to(device)
            gt = gt.to(device)
            corresponding_bbox = corresponding_bbox.to(device)
            centroid = centroid.to(device)
            m = m.to(device)

            initial_inds = torch.unsqueeze(torch.arange(start=0, end=data.shape[1]), 0).repeat(data.shape[0], 1).to(
                device)
            aggregated_xyz, vote_xyz, result, seed_inds, vote_inds = model(data, initial_inds)
            aggregated_xyz = aggregated_xyz.transpose(1, 2)
            gt_object = torch.gather(corresponding_bbox, 1, vote_inds.to(torch.long)).to(torch.long).to(device)
            vote_mask = (gt_object >= 0).byte().to(device)
            gt_object[gt_object == -1000] = 0
            gt_vote = torch.gather(gt[:, :, 12:15], 1, gt_object[:, :, None].repeat(1, 1, 3))
            vote_xyz = vote_xyz * m[:, None, None]
            result[:, 1:7, :] = result[:, 1:7, :] * m[:, None, None]
            aggregated_xyz = aggregated_xyz * m[:, None, None]
            vote_xyz = vote_xyz + centroid[:, None, :]
            result[:, 1:4, :] = result[:, 1:4, :] + centroid[:, :, None]
            aggregated_xyz = aggregated_xyz + centroid[:, None, :]

            M_pos = torch.sum(vote_mask, 1) + 1e-6
            vote_loss = torch.sum((torch.sum(torch.abs(vote_xyz - gt_vote), dim=2) * vote_mask), 1) / M_pos
            vote_loss = torch.mean(vote_loss)

            dist1, ind1, _, _ = nn_distance(aggregated_xyz, gt[:, :, 12:15])
            euclidean_dist1 = torch.sqrt(dist1 + 1e-6)
            """
                # objectness_label: 1 if pred object center is within NEAR_THRESHOLD of any GT object
                # objectness_mask: 0 if pred object center is in gray zone (DONOTCARE), 1 otherwise
            """

            objectness_label = torch.zeros(aggregated_xyz.shape[0], aggregated_xyz.shape[1], dtype=torch.long).to(device)
            objectness_mask = torch.zeros(aggregated_xyz.shape[0], aggregated_xyz.shape[1]).to(device)
            object_assignment = ind1

            pred_car_prob = 1 / (1 + torch.exp(-result[:, 0, :]))
            objectness_label[pred_car_prob > 0.9995] = 1
            objectness_mask[pred_car_prob > 0.9995] = 1

            print("positive preds: ", torch.sum(objectness_label))
            print("negative preds: ", torch.sum(objectness_mask) - torch.sum(objectness_label))

            center_loss, size_loss, car_prob_loss = loss.compute_box_loss(gt, objectness_label, objectness_mask,
                                                                          object_assignment, result.transpose(1, 2))
            print("vote loss: ", vote_loss)
            print("center loss: ", center_loss)
            print("size loss: ", size_loss)
            print("car prob loss: ", car_prob_loss)

            vote_loss_values.append(vote_loss)
            center_loss_values.append(center_loss)
            size_loss_values.append(size_loss)
            car_prob_loss_values.append(car_prob_loss)
            loss1 = vote_loss + center_loss + size_loss + car_prob_loss

            total_loss_values.append(loss1)

            print("total loss ", loss1)

        # plt.plot(vote_loss_values, 'b', label='Vote Loss')
        plt.plot(center_loss_values, color = 'red',marker = 'o' , label='Center Loss')
        plt.plot(size_loss_values, 'g', label='Size Loss')
        # plt.plot(total_loss_values, 'm', label='Total Loss')
        plt.legend(framealpha=1, frameon=True)
        plt.savefig("test_loss_plot.png")
        print("vote loss: " , sum(vote_loss_values) / len(vote_loss_values))
        print("center loss: " , sum(center_loss_values) / len(center_loss_values))
        print("size loss: " ,sum(size_loss_values) / len(size_loss_values))
        print("car_prob_loss: ", sum(car_prob_loss_values) / len(car_prob_loss_values))
        print("total loss: " , sum(total_loss_values) / len(total_loss_values))