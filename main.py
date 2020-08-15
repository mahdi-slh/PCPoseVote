import torch
import numpy as np
import torch.utils.data as torch_data
from kitty import KittyDataset
import torch.optim as optim
import loss
from config import *
from Network import Votenet
from tqdm import tqdm
import matplotlib.pyplot as plt

if __name__ == '__main__':
    torch.manual_seed(0)
    np.random.seed(0)
    torch.multiprocessing.freeze_support()

    train_set = KittyDataset(PATH)
    train_loader = torch_data.DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=1)
    model = Votenet(num_class=2, num_heading_bin=2, num_size_cluster=2, mean_size_arr=np.zeros((3, 1)),
                    input_feature_dim=2)
    model.to(device)
    torch.multiprocessing.freeze_support()
    optimizer = optim.Adam(model.parameters(), lr=initial_learning_rate)

    # checkpoint = torch.load(SAVE_PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    epoch = 0

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    center_loss_values = []
    vote_loss_values = []
    size_loss_values = []
    total_loss_values = []

    r = 0
    plt.plot(vote_loss_values, 'b', label='Vote Loss')
    plt.plot(center_loss_values, 'r', label='Center Loss')
    plt.plot(size_loss_values, 'g', label='Size Loss')
    plt.plot(total_loss_values, 'm', label='Total Loss')
    plt.legend(framealpha=1, frameon=True)

    for epoch in tqdm(range(epoch + 1, epoch + epoch_num)):
        r = r + 1
        plt.plot(vote_loss_values, 'b', label='Vote Loss')
        plt.plot(center_loss_values, 'r', label='Center Loss')
        plt.plot(size_loss_values, 'g', label='Size Loss')
        plt.plot(total_loss_values, 'm', label='Total Loss')
        plt.savefig("loss_plot.png")

        if r == 1:
            plt.close()
            r = 0
            center_loss_values.clear()
            vote_loss_values.clear()
            size_loss_values.clear()
            total_loss_values.clear()
            # plt.plot(vote_loss_values, 'b', label='Vote Loss')
            plt.plot(center_loss_values, 'r', label='Center Loss')
            plt.plot(size_loss_values, 'g', label='Size Loss')
            plt.plot(total_loss_values, 'm', label='Total Loss')
            plt.legend(framealpha=1, frameon=True)


        for i, data in tqdm(enumerate(train_loader, 0)):
            optimizer.zero_grad()
            print("epoch ", epoch, " step ", i + 1)

            data, gt, corresponding_bbox, centroid, m = data
            # data, gt, corresponding_bbox = data

            data = data.to(torch.float32)

            data = data.to(device)
            gt = gt.to(device)
            centroid = centroid.to(device)
            m = m.to(device)

            initial_inds = torch.unsqueeze(torch.arange(start=0, end=data.shape[1]), 0).repeat(data.shape[0], 1)

            l1_xyz, vote_xyz, aggregation, seed_inds = model(data, initial_inds)

            l1_xyz = l1_xyz * m[:, None, None]
            vote_xyz = vote_xyz * m[:, None, None]
            aggregation[:, 1:7, :] = aggregation[:, 1:7, :] * m[:, None, None]
            l1_xyz = l1_xyz + centroid[:, :, None]
            vote_xyz = vote_xyz + centroid[:, None, :]
            aggregation[:, 1:4, :] = aggregation[:, 1:4, :] + centroid[:, :, None]

            # forward + backward + optimize
            end_point_locs = torch.zeros(batch_size, num_proposal, 3).to(device)

            velodyne_center = torch.mean(data, dim=1)
            velodyne_center = velodyne_center[:, None, :].repeat(1,num_proposal,1)
            for i in range(batch_size):
                # end_point_locs[i] = data[i, seed_inds[i]]
                end_point_locs[i] = velodyne_center[i]

            vote_loss = loss.compute_vote_loss(gt, l1_xyz, vote_xyz, train_set)

            center_loss, size_loss, angle_loss = loss.compute_box_loss(gt, corresponding_bbox, aggregation, seed_inds,
                                                                       train_set, end_point_locs)
            print("vote loss: ", vote_loss)
            print("center loss: ", center_loss)
            print("size loss: ", size_loss)

            vote_loss_values.append(vote_loss)
            center_loss_values.append(center_loss)
            size_loss_values.append(size_loss)
            # print("angle loss: ", angle_loss)
            loss1 = vote_loss + size_loss + center_loss

            total_loss_values.append(loss1)

            print("total loss ", loss1)
            loss1.backward()

            optimizer.step()

            scheduler.step()
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss1,
            }, SAVE_PATH)
