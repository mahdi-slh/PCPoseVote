import torch
import numpy as np
import torch.utils.data as torch_data
from kitty import KittyDataset
import torch.optim as optim
import loss
from config import *
from Network import Votenet

if __name__ == '__main__':

    torch.multiprocessing.freeze_support()

    train_set = KittyDataset(PATH)
    train_loader = torch_data.DataLoader(train_set, shuffle=True, batch_size=batch_size, num_workers=1)
    model = Votenet(num_class=2, num_heading_bin=2, num_size_cluster=2, mean_size_arr=np.zeros((3, 1)),
                    input_feature_dim=2)
    model.to(device)
    torch.multiprocessing.freeze_support()
    optimizer = optim.SGD(model.parameters(), lr=0.0001, momentum=0.9)

    # checkpoint = torch.load(SAVE_PATH)
    # model.load_state_dict(checkpoint['model_state_dict'])
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    # epoch = checkpoint['epoch']
    epoch = 0

    for epoch in range(epoch + 1, epoch + epoch_num + 1):
        for i, data in enumerate(train_loader, 0):
            print("epoch ", epoch, " step ", i + 1)
            data, gt, corresponding_bbox = data

            data = data.to(torch.float32)
            data = data.to(device)
            gt = gt.to(device)

            initial_inds = torch.unsqueeze(torch.arange(start=0, end=data.shape[1]), 0).repeat(data.shape[0], 1)

            l1_xyz, vote_xyz, aggregation, seed_inds = model(data, initial_inds)

            optimizer.zero_grad()
            # forward + backward + optimize
            vote_loss = loss.compute_vote_loss(gt, l1_xyz, vote_xyz, train_set)
            center_loss, size_loss, angle_loss = loss.compute_box_loss(gt, corresponding_bbox, aggregation, seed_inds,
                                                                       train_set)
            print("vote loss: ", vote_loss)
            print("center loss: ", center_loss)
            print("size loss: ", size_loss)
            # print("angle loss: ", angle_loss)
            loss1 = vote_loss + size_loss + center_loss
            print("total loss ", loss1)
            loss1.backward()
            optimizer.step()

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss1,
            }, SAVE_PATH)
