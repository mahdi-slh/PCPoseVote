import torch.nn as nn
from config import *
from Pointnet2 import *


class VotingModule(nn.Module):
  def __init__(self, vote_factor, seed_feature_dim):
    """ Votes generation from seed point features.

    Args:
        vote_facotr: int
            number of votes generated from each seed point
        seed_feature_dim: int
            number of channels of seed point features
    """
    super().__init__()
    self.vote_factor = vote_factor
    self.in_dim = seed_feature_dim
    self.out_dim = self.in_dim  # due to residual feature, in_dim has to be == out_dim
    self.conv1 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
    self.conv2 = torch.nn.Conv1d(self.in_dim, self.in_dim, 1)
    self.conv3 = torch.nn.Conv1d(self.in_dim, (3 + self.out_dim) * self.vote_factor, 1)
    self.bn1 = torch.nn.BatchNorm1d(self.in_dim)
    self.bn2 = torch.nn.BatchNorm1d(self.in_dim)

  def forward(self, seed_xyz, seed_features):
    """ Forward pass.

    Arguments:
        seed_xyz: (batch_size, num_seed, 3) Pytorch tensor
        seed_features: (batch_size, feature_dim, num_seed) Pytorch tensor
    Returns:
        vote_xyz: (batch_size, num_seed*vote_factor, 3)
        vote_features: (batch_size, vote_feature_dim, num_seed*vote_factor)
    """

    num_seed = seed_xyz.shape[1]
    num_vote = num_seed * self.vote_factor
    net = F.relu(self.bn1(self.conv1(seed_features)))
    net = F.relu(self.bn2(self.conv2(net)))
    net = self.conv3(net)  # (batch_size, (3+out_dim)*vote_factor, num_seed)
    net = net.transpose(2, 1).view(seed_xyz.shape[0], num_seed, self.vote_factor, 3 + self.out_dim)
    offset = net[:, :, :, 0:3]
    vote_xyz = seed_xyz.unsqueeze(2) + offset
    vote_xyz = vote_xyz.contiguous().view(seed_xyz.shape[0], num_vote, 3)

    residual_features = net[:, :, :, 3:]  # (batch_size, num_seed, vote_factor, out_dim)
    vote_features = seed_features.transpose(2, 1).unsqueeze(2) + residual_features
    vote_features = vote_features.contiguous().view(seed_xyz.shape[0], num_vote, self.out_dim)
    vote_features = vote_features.transpose(2, 1).contiguous()

    return vote_xyz, vote_features


class ProposalModule(nn.Module):
  def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr, num_proposal, sampling, in_channel,
               seed_feat_dim=256):
    super().__init__()

    self.num_class = num_class
    self.num_heading_bin = num_heading_bin
    self.num_size_cluster = num_size_cluster
    self.mean_size_arr = mean_size_arr
    self.num_proposal = num_proposal
    self.sampling = sampling
    self.seed_feat_dim = seed_feat_dim
    self.in_channel = in_channel

    # Vote clustering
    self.vote_aggregation = PointNetSetAbstraction(
      npoint=self.num_proposal,
      radius=0.3,
      nsample=64,
      in_channel=self.in_channel,
      mlp=[self.in_channel, 128, 128, 128],
      group_all=False
    )

    self.conv1 = torch.nn.Conv1d(128, 128, 1)
    self.conv2 = torch.nn.Conv1d(128, 128, 1)
    self.conv3 = torch.nn.Conv1d(128, 1 + 3 + 3 + 1, 1)  # car prob + center loc + size + angle
    self.bn1 = torch.nn.BatchNorm1d(128)
    self.bn2 = torch.nn.BatchNorm1d(128)

  def forward(self, xyz, features, seed_inds):
    # Farthest point sampling (FPS) on votes
    new_xyz, new_features, seed_inds = self.vote_aggregation(xyz, features, seed_inds)

    # --------- PROPOSAL GENERATION ---------
    net = F.relu(self.bn1(self.conv1(new_features)))
    net = F.relu(self.bn2(self.conv2(net)))
    net = self.conv3(net)  # (batch_size, 2+3+num_heading_bin*2+num_size_cluster*4, num_proposal)
    net[:, 1:4, :] = new_xyz + net[:,1:4,:]
    return net, seed_inds


class Votenet(nn.Module):
  def __init__(self, num_class, num_heading_bin, num_size_cluster, mean_size_arr,
               input_feature_dim=0, num_proposal=num_proposal, vote_factor=1, sampling='vote_fps'):
    self.num_class = num_class
    self.num_heading_bin = num_heading_bin
    self.num_size_cluster = num_size_cluster
    self.mean_size_arr = mean_size_arr
    self.input_feature_dim = input_feature_dim
    self.num_proposal = num_proposal
    self.vote_factor = vote_factor
    self.sampling = sampling

    super(Votenet, self).__init__()
    in_channel = 3

    self.sa1 = PointNetSetAbstraction(npoint=256, radius=0.2, nsample=64, in_channel=in_channel,
                                      mlp=[64, 64, 128], group_all=False)
    self.sa2 = PointNetSetAbstraction(npoint=128, radius=0.4, nsample=64, in_channel=128 + 3,
                                      mlp=[128, 128, 256], group_all=False)
    self.sa3 = PointNetSetAbstraction(npoint=64, radius=0.8, nsample=64, in_channel=256 + 3,
                                      mlp=[128, 128, 256], group_all=False)
    self.sa4 = PointNetSetAbstraction(npoint=64, radius=0.8, nsample=64, in_channel=256 + 3,
                                      mlp=[128, 128, 256], group_all=False)
    self.fp4 = PointNetFeaturePropagation(in_channel=512, mlp=[256, 256])
    self.fp3 = PointNetFeaturePropagation(in_channel=512, mlp=[256, 256])
    self.fp2 = PointNetFeaturePropagation(in_channel=384, mlp=[256, seed_features_dim])
    self.vgen = VotingModule(self.vote_factor, seed_features_dim)
    self.pnet = ProposalModule(num_class, num_heading_bin, num_size_cluster,
                               mean_size_arr, num_proposal, sampling, in_channel=3 + seed_features_dim)

  def forward(self, xyz, seed_inds):
    # Set Abstraction layers

    xyz = xyz.permute(0, 2, 1)

    l0_xyz = xyz
    l1_xyz, l1_points, seed_inds = self.sa1(l0_xyz, None, seed_inds)
    l2_xyz, l2_points, _ = self.sa2(l1_xyz, l1_points, seed_inds)
    l3_xyz, l3_points, _ = self.sa3(l2_xyz, l2_points, seed_inds)
    l4_xyz, l4_points, _ = self.sa4(l3_xyz, l3_points, seed_inds)

    # Feature Propagation layers

    l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
    l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
    l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)

    vote_xyz, features = self.vgen(l1_xyz.permute(0, 2, 1), l1_points)
    # features_norm = torch.norm(features, p=2, dim=1)
    # features = features.div(features_norm.unsqueeze(1))
    """
        xyz.shape = [B,3,l1_xyz.npoint]
        features.shape = [B,seed_features_dim,l1_xyz.npoint]
    """

    end_points, seed_inds = self.pnet(torch.transpose(vote_xyz, 1, 2), features, seed_inds)
    return l1_xyz, vote_xyz, end_points, seed_inds
