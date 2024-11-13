import torch.nn as nn
import torch
import numpy as np

def get_confusion_mat(pred,label,cls):
    pred_is_cls = (pred==cls)
    num_curr_cls_pred = (pred_is_cls*1).sum().item()
    y_is_cls = (label==cls)
    total_curr_cls = (y_is_cls*1).sum().item()
    total_match_cls = (torch.logical_and(pred_is_cls, y_is_cls)*1).sum().item()
    return (2*total_match_cls)/(total_curr_cls+num_curr_cls_pred)

def micro_dice(predictions, labels, num_classes=19):
    predictions = torch.reshape(predictions, (-1,))
    labels = torch.reshape(labels, (-1,))
    #print('predictions', predictions.shape)
    #print('labels', labels.shape)
    predictions = cp.asarray(predictions)
    labels = cp.asarray(labels)

    # Initialize TP, FP, and FN arrays for each class
    TP = cp.zeros(num_classes, dtype=cp.int32)
    FP = cp.zeros(num_classes, dtype=cp.int32)
    FN = cp.zeros(num_classes, dtype=cp.int32)

    for cls in range(num_classes):
        # Calculate TP, FP, and FN for each class in parallel
        TP[cls] = cp.sum((predictions == cls) & (labels == cls))
        FP[cls] = cp.sum((predictions == cls) & (labels != cls))
        FN[cls] = cp.sum((predictions != cls) & (labels == cls))

    # Sum up TP, FP, and FN across all classes
    TP_total = TP.sum()
    FP_total = FP.sum()
    FN_total = FN.sum()

    # Calculate the micro-average Dice score
    dice_score = (2 * TP_total) / (2 * TP_total + FP_total + FN_total) if (2 * TP_total + FP_total + FN_total) > 0 else 0.0

    # Return result to CPU memory as a Python float
    return float(dice_score)
def macro_dice(predictions, labels, num_classes=19):
    num_cls = num_classes
    predictions = torch.reshape(predictions, (-1,)).numpy()
    labels = torch.reshape(labels, (-1,)).numpy()
    #print('predictions', predictions.shape)
    #print('labels', labels.shape)
    #predictions = cp.asarray(predictions)
    #labels = cp.asarray(labels)

    # Initialize TP, FP, and FN arrays for each class
    TP = torch.zeros(num_classes)
    FP = torch.zeros(num_classes)
    FN = torch.zeros(num_classes)
    dice_avg=0
    for i in range(num_classes):
        denorm = (np.sum(predictions[(predictions==i)]==i) + np.sum(labels[labels==i]==i))
        if denorm==0:
            num_cls-=1
            continue
        d = np.sum(predictions[(labels==i)]==i)*2.0 / denorm
        dice_avg+=d
    return dice_avg/num_cls


def reset_lr(optimizer, lr_ratio = 0.5):
    for g in optimizer.param_groups:
        g['lr'] = g['lr']*lr_ratio
    return optimizer
def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.
    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist
def timeit(tag, t):
    print("{}: {}s".format(tag, time() - t))
    return time()

def pc_normalize(pc):
    l = pc.shape[0]
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m
    return pc
def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
        points: input points data, [B, S, C]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points
def farthest_point_sample(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids

def query_ball_point_custom(radius, nsample, point, node):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        point: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = point.device
    B, N, C = point.shape
    _, S, _ = node.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(node, point)
    group_idx[sqrdists > radius ** 2] = N-1#if distance larger than radius,point index set to 5999
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = (group_idx == N-1)
    group_idx[mask] = group_first[mask]
    return group_idx

def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx
    
def sample_and_group(npoint, radius, nsample, xyz, points, returnfps=False):
    B, N, C = xyz.shape
    S = npoint
    fps_idx = farthest_point_sample(xyz, npoint) # [B, npoint, C]
    new_xyz = index_points(xyz, fps_idx)
    idx = query_ball_point(radius, nsample, xyz, new_xyz)
    grouped_xyz = index_points(xyz, idx) # [B, npoint, nsample, C]
    grouped_xyz_norm = grouped_xyz - new_xyz.view(B, S, 1, C)

    if points is not None:
        grouped_points = index_points(points, idx)
        new_points = torch.cat([grouped_xyz_norm, grouped_points], dim=-1) # [B, npoint, nsample, C+D]
    else:
        new_points = grouped_xyz_norm
    if returnfps:
        return new_xyz, new_points, grouped_xyz, fps_idx
    else:
        return new_xyz, new_points


def sample_and_group_all(xyz, points):
    device = xyz.device
    B, N, C = xyz.shape
    new_xyz = torch.zeros(B, 1, C).to(device)
    grouped_xyz = xyz.view(B, 1, N, C)
    if points is not None:
        new_points = torch.cat([grouped_xyz, points.view(B, 1, N, -1)], dim=-1)
    else:
        new_points = grouped_xyz
    return new_xyz, new_points