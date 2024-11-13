import torch.nn as nn
import model.point_graph_func as pg_func
from model.utilities import *
import torch.nn.functional as F
class PointNetSetAbstraction(nn.Module):
    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel
        self.group_all = group_all

    def forward(self, xyz, points):
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)

        if self.group_all:
            new_xyz, new_points = sample_and_group_all(xyz, points)
        else:
            new_xyz, new_points = sample_and_group(self.npoint, self.radius, self.nsample, xyz, points)
        new_points = new_points.permute(0, 3, 2, 1) # [B, C+D, nsample,npoint]
        for i, conv in enumerate(self.mlp_convs):
            bn = self.mlp_bns[i]
            new_points =  F.relu(bn(conv(new_points)))

        new_points = torch.max(new_points, 2)[0]
        new_xyz = new_xyz.permute(0, 2, 1)
        return new_xyz, new_points
class Pointnet2(nn.Module):
    def __init__(self, specs):
        super(Pointnet2, self).__init__()
        pointAbstractions = specs["PointAbstractions"]
        num_classes = specs["Out_Channel_Dim"]
        self.sa1 = PointNetSetAbstraction(pointAbstractions[0], 0.1, 32, 3+3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(pointAbstractions[1], 0.2, 32, 64+3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(pointAbstractions[2], 0.4, 32, 128 + 3, [128, 128, 256], False)
        self.sa4 = PointNetSetAbstraction(pointAbstractions[3], 0.8, 32, 256 + 3, [256, 256, 512], False)
        self.fp4 = pg_func.PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = pg_func.PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = pg_func.PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = pg_func.PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, num_classes, 1)
        self.bn1 = nn.BatchNorm1d(num_classes)
        self.drop1 = nn.Dropout(0.1)
        #self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)
        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)    
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)    
        x = F.relu(self.bn1(self.conv1(l0_points)))
        x = x.permute(0, 2, 1)
        return x, l0_points
