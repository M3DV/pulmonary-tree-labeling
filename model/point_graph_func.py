import torch.nn as nn
import torch
from model.utilities import *
import torch.nn.functional as F

class PGMLP(nn.Module):
    def __init__(self, in_channel, hidden):
        super(PGMLP, self).__init__()
        """
        Initialize the Point-Graph Fusion Layer, which takes input feature from both the point-based and graph-based branch, 
        and produce corresponding point-based and graph-based updated feature.
    
        Args:
        in_channel: 
        hidden: 
        """
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in hidden:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel

    def forward(self,points):
        """
        Input:
            points: input points position data, [B, C, N]
        Return:
            points_feature: points feature data, [B, C_1, N]
        """
        B,C,N = points.shape
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            points =  F.relu(bn(conv(points)))
        return points.permute(0,2,1)

class merge_mlp_layer(nn.Module):
    def __init__(self, in_channel, hidden,merge_method='concat',is_final=False):
        super(merge_mlp_layer, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.is_final = is_final
        last_channel = in_channel
        if merge_method=='sum':
            self.bn1 = nn.BatchNorm1d(in_channel)
            self.bn2 = nn.BatchNorm1d(in_channel)
        else:
            self.bn1 = nn.BatchNorm1d(int(in_channel/2))
            self.bn2 = nn.BatchNorm1d(int(in_channel/2))
        for out_channel in hidden:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.method = merge_method
    def forward(self,fea1,fea2):
        """
        Input:
            fea1: input points position data, [B, C, N]
            fea2: input points position data, [B, C, N]
        Return:
            points_feature: points feature data, [B, C_1, N]
        """
        B,C,N = fea1.shape
        if self.method=='sum':
            feature = fea1+fea2
        else:
            feature = torch.cat((fea1,fea2),dim=1)
        for i, conv in enumerate(self.mlp_convs):
            if i==(len(self.mlp_convs)-1) and self.is_final:
                feature = conv(feature)
            else:
                bn = self.mlp_bns[i]
                feature =  F.relu(bn(conv(feature)))
        return feature

class Grouping_MLP_MP(nn.Module):
    def __init__(self,in_channel,hidden_channel):
        super(Grouping_MLP_MP, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.relu = nn.ReLU(inplace=True)
        last_channel = in_channel
        for out_channel in hidden_channel:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
    def forward(self,points):
        points = points.permute(0, 3, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            points =  self.relu(bn(conv(points)))
        points = torch.max(points, 2)[0]
        return points.permute(0,2,1)

class classification_head(nn.Module):
    def __init__(self,in_channel,hidden_channel,num_class):
        super(classification_head, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        hidden_channel=hidden_channel
        for out_channel in hidden_channel:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.final = nn.Conv1d(last_channel, num_class, 1)
        self.relu = nn.ReLU(inplace=True)
    def forward(self,points):
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            points =  self.relu(bn(conv(points)))
        points = self.final(points)
        return points

class PointNetFeaturePropagation(nn.Module):
    def __init__(self, in_channel, mlp, fp_num=3):
        super(PointNetFeaturePropagation, self).__init__()
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.fp_num = fp_num
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv1d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm1d(out_channel))
            last_channel = out_channel
        self.relu = nn.ReLU(inplace=True)
    def forward(self, xyz1, xyz2, points1, points2):
        '''
        Expected Input Shape:
        xyz1 = (B,C,N)
        xyz2 = (B,C,S)

        points2 = (B,Feature,S)
        '''
        xyz1 = xyz1.permute(0, 2, 1)
        xyz2 = xyz2.permute(0, 2, 1)

        points2 = points2.permute(0, 2, 1)
        B, N, C = xyz1.shape
        _, S, _ = xyz2.shape

        if S == 1:
            interpolated_points = points2.repeat(1, N, 1)
        else:
            dists = square_distance(xyz1, xyz2)
            dists, idx = dists.sort(dim=-1)
            dists, idx = dists[:, :, :self.fp_num], idx[:, :, :self.fp_num]  # [B, N, 3]

            dist_recip = 1.0 / (dists + 1e-8)
            norm = torch.sum(dist_recip, dim=2, keepdim=True)
            weight = dist_recip / norm
            interpolated_points = torch.sum(index_points(points2, idx) * weight.view(B, N, self.fp_num, 1), dim=2)

        if points1 is not None:
            points1 = points1.permute(0, 2, 1)
            new_points = torch.cat([points1, interpolated_points], dim=-1)
        else:
            new_points = interpolated_points

        new_points = new_points.permute(0, 2, 1)
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points = self.relu(bn(conv(new_points)))
        return new_points
