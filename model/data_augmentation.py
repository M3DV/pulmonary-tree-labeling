import numpy as np
import torch
def rotate_perturbation_point_cloud_g(graph, angle_sigma=0.6, angle_clip=0.18):
    '''
    perform random rotation of the point cloud and graph skeleton together.
    '''
    all_nodes = graph.x.numpy().reshape(-1)
    pos_mask = (all_nodes!=-10)
    neg_mask = (all_nodes==-10)
    nodes = all_nodes[pos_mask].reshape(-1,3)
    pad_nodes = all_nodes[neg_mask].reshape(-1,3)
    points = graph.points
    points_inf = graph.implicit_points
    #concat all graph nodes, point cloud points, and query points together
    data = np.concatenate((nodes, points, points_inf), axis=0)
    #perform augmentation
    rotated_data = np.zeros(data.shape, dtype=np.float32)
    angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
    R = np.dot(Rz, np.dot(Ry,Rx))
    shape_pc = data
    rotated_data = np.dot(shape_pc.reshape((-1, 3)), R)
    results = torch.from_numpy(rotated_data)
    #reassign the augmented 3D coordinates back to the graph data
    graph.x = torch.from_numpy(np.concatenate((results[:nodes.shape[0]],pad_nodes),axis=0))
    graph.points = results[nodes.shape[0]:nodes.shape[0]+points.shape[0]].numpy()
    graph.implicit_points = results[nodes.shape[0]+points.shape[0]:].numpy()
    return graph
def shift_point_cloud_g(graph, shift_range=0.1):
    '''
    perform random shifting of the point cloud and graph skeleton together.
    '''
    all_nodes = graph.x.numpy().reshape(-1)
    pos_mask = (all_nodes!=-10)
    neg_mask = (all_nodes==-10)
    #seperate real nodes and padding nodes
    nodes = all_nodes[pos_mask].reshape(-1,3)
    pad_nodes = all_nodes[neg_mask].reshape(-1,3)

    points = graph.points
    points_inf = graph.implicit_points
    #concat all graph nodes, point cloud points, and query points together
    batch_data = np.concatenate((nodes, points,points_inf), axis=0)
    #perform augmentation
    N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (3))
    batch_data += shifts
    results = torch.from_numpy(batch_data)
    #reassign the augmented 3D coordinates back to the graph data
    graph.x = torch.from_numpy(np.concatenate((results[:nodes.shape[0]],pad_nodes),axis=0))
    graph.points = results[nodes.shape[0]:nodes.shape[0]+points.shape[0]].numpy()
    graph.implicit_points = results[nodes.shape[0]+points.shape[0]:].numpy()
    return graph
def random_scale_point_cloud_g(graph, scale_low=0.9, scale_high=1.15):
    '''
    perform random scaling of the point cloud and graph skeleton together.
    '''
    all_nodes = graph.x.numpy().reshape(-1)
    pos_mask = (all_nodes!=-10)
    neg_mask = (all_nodes==-10)
    #seperate real nodes and padding nodes
    nodes = all_nodes[pos_mask].reshape(-1,3)
    pad_nodes = all_nodes[neg_mask].reshape(-1,3)

    points = graph.points
    points_inf = graph.implicit_points

    #concat all graph nodes, point cloud points, and query points together
    batch_data = np.concatenate((nodes, points,points_inf), axis=0)
    N, C = batch_data.shape

    #perform augmentation
    scales = np.random.uniform(scale_low, scale_high, 1)
    batch_data *= scales
    results = torch.from_numpy(batch_data)

    #reassign the augmented 3D coordinates back to the graph data
    graph.x = torch.from_numpy(np.concatenate((results[:nodes.shape[0]],pad_nodes),axis=0))
    graph.points = results[nodes.shape[0]:nodes.shape[0]+6000].numpy()
    graph.implicit_points = results[nodes.shape[0]+6000:].numpy()
    return graph

def minmax_norm_g(graph):
    '''
    perform min-max normalization on the point cloud and graph skeleton together.
    '''
    all_nodes = graph.x.numpy().reshape(-1)
    pos_mask = (all_nodes!=-10)
    neg_mask = (all_nodes==-10)
    #seperate real nodes and padding nodes
    nodes = all_nodes[pos_mask].reshape(-1,3)
    pad_nodes = all_nodes[neg_mask].reshape(-1,3)

    points = graph.points
    points_inf = graph.implicit_points

    #concat all graph nodes, point cloud points, and query points together
    x = np.concatenate((nodes, points,points_inf), axis=0)
    
    #perform augmentation
    maxs = np.amax(x,axis=0)
    mins = np.amin(x,axis=0)
    x_shifted = (x-mins)
    x = x_shifted/(maxs-mins)
    results = torch.from_numpy(x)

    #reassign the augmented 3D coordinates back to the graph data
    graph.x = torch.from_numpy(np.concatenate((results[:nodes.shape[0]],pad_nodes),axis=0))
    graph.points = results[nodes.shape[0]:nodes.shape[0]+points.shape[0]].numpy()
    graph.implicit_points = results[nodes.shape[0]+points.shape[0]:].numpy()
    return graph