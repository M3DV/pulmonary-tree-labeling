import os 
import copy
import numpy as np
import torch
from torch import optim
from model.utilities import *
import model.data_augmentation as aug
from torch_geometric.loader import DataLoader as pygeo_dataloader
from model.pygeo_dataset import PultreeDataset, get_pultree_dataloader
import torch_geometric.transforms as T_geometric
import nibabel as nib

def PGN_train_step(batch, model, optimizer, criterion, is_train=True, device = None):
    optimizer.zero_grad()  # Clear gradients.
    total_pts,correct_pts,total_nodes,correct_nodes = 0,0,0,0
    if is_train:
        model.train()
    else:
        model.eval()
    label = torch.stack([torch.from_numpy(l) for l in batch.points_label],dim=0).long().to(device)
    ei = batch.edge_index.to(device)
    pts = torch.stack([torch.from_numpy(pt) for pt in batch.points],dim=0).float().to(device)#[8, 6000, 3]
    B,N,C = pts.shape
    nodes = batch.x.view(B,-1,C).float().to(device)#[8, 519, 3]
    y_use = ((batch.y[batch.node_mask])-1).detach().cpu()
    y_use_mask = (y_use!=-11)
    y_use = y_use[y_use_mask]
    y_use_onehot = torch.nn.functional.one_hot(y_use.long()).float().to(device)
    node_loss,pt_loss = 0,0
    if is_train:
        points_initial_feature, pt_outs, node_outs, p_features, n_features = model(pts,nodes, ei, is_train = True)
        for pt_out,node_out in zip(pt_outs,node_outs):
            pred_use_onehot = (node_out[batch.node_mask])[y_use_mask]
            node_loss += criterion(pred_use_onehot, y_use_onehot.float())
            pt_loss += criterion(pt_out, label)#use this for ce loss only
        total_loss = pt_loss+node_loss
    else:
        with torch.no_grad():
            points_initial_feature, pt_outs, node_outs, p_features, n_features = model(pts,nodes, ei, is_train = False)
            for pt_out,node_out in zip(pt_outs,node_outs):
                pred_use_onehot = (node_out[batch.node_mask])[y_use_mask]
                node_loss += criterion(pred_use_onehot, y_use_onehot.float())
                pt_loss = criterion(pt_out, label)#use this for ce loss only
            total_loss = pt_loss+node_loss
    dice_avg = 0

    if is_train:
        total_loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
    l = total_loss.detach().item()

    with torch.no_grad():
        pred_use = torch.argmax(pt_outs[-1], dim=1).detach().to('cpu')
        pred_dice = pred_use.numpy()[0]
        label_dice = label.to('cpu').numpy()[0]
        
        correct_pts += pred_use.eq(label.to('cpu')).sum().item()
        total_pts += ((label!=-2)*1).sum().item()
        
        pred_use_onehot = (node_outs[-1][batch.node_mask])[y_use_mask]
        pred_use_node = torch.argmax(pred_use_onehot, dim=1).detach().to('cpu')

        graph_dice = macro_dice(pred_use_node,y_use.to('cpu'),19)
        point_dice = macro_dice(pred_use,label.to('cpu'),19)

        correct_nodes += pred_use_node.eq(y_use.to('cpu')).sum().item()
        total_nodes += ((y_use>-1000)*1).sum().item()
    del total_loss, y_use_onehot,label, ei, pts, points_initial_feature, pt_outs, node_outs, p_features, n_features
    return (pt_loss, correct_pts, total_pts, point_dice), (node_loss, correct_nodes, total_nodes, graph_dice)    
def PGN_run_epoch(loader, model, optimizer, criterion, is_train=True,device = None):
    total_pts = 0
    total_pts_correct = 0
    final_pt_loss = 0
    final_pt_dice = 0
    total_nodes = 0
    total_nodes_correct = 0
    final_node_loss = 0
    final_node_dice = 0
    for batch_i, batch in enumerate(loader):
        point_feedback, graph_feedback = PGN_train_step(batch, model, optimizer, criterion, is_train=is_train, device = device)
        pt_loss,pts_correct,pts_num, pt_dice = point_feedback
        node_loss,nodes_correct,nodes_num, node_dice = graph_feedback

        total_pts_correct+=pts_correct
        total_pts+=pts_num
        final_pt_loss+=pt_loss*(len(batch.ptr)-1)
        final_pt_dice+=pt_dice*(len(batch.ptr)-1)
        total_nodes_correct+=nodes_correct
        total_nodes+=nodes_num
        final_node_loss+=node_loss*(len(batch.ptr)-1)
        final_node_dice+=node_dice*(len(batch.ptr)-1)

    pts_accuracy = total_pts_correct/total_pts
    final_pt_loss /= len(loader.dataset)
    final_pt_dice /= len(loader.dataset)

    nodes_accuracy = total_nodes_correct/total_nodes
    final_node_loss /= len(loader.dataset)
    final_node_dice /= len(loader.dataset)
    return (pts_accuracy, final_pt_loss, final_pt_dice), (nodes_accuracy, final_node_loss, final_node_dice)
def point_graph_network_train(specs, ds_specs, PGN, device = None, base_dir = '/'):
    point_encoder_path = os.path.join(base_dir, ds_specs["Model_Save_Path"], specs["Point_Training_Specs"]["checkpoint_filename"])
    graph_encoder_path = os.path.join(base_dir, ds_specs["Model_Save_Path"], specs["Graph_Training_Specs"]["checkpoint_filename"])
    PGN.point_encoder.load_state_dict(torch.load(point_encoder_path))
    PGN.graph_encoder.load_state_dict(torch.load(graph_encoder_path))
    print('initial encoders loaded')
    point_graph_network_path = os.path.join(base_dir, ds_specs["Model_Save_Path"], 'Point_Graph_Network')

    load_model = (specs["PGN_Training_Specs"]["load_Point_Graph_Network"]==1)
    save_model = (specs["PGN_Training_Specs"]["save_Point_Graph_Network"]==1)
    
    epoches = specs["PGN_Training_Specs"]["Num_Epoches"]
    batch_size = specs["PGN_Training_Specs"]["Batch_Size"]
    learning_rate = specs["PGN_Training_Specs"]["Learning_Rate"]
    lr_decay_ratio = specs["PGN_Training_Specs"]["Learning_Rate_Decay"]
    lr_decay_epoch = specs["PGN_Training_Specs"]["Lr_Decay_Epoches"]
    val_per_epoch = specs["PGN_Training_Specs"]["Validation_Per_Epoches"]
    Snapshot_Frequency = specs["PGN_Training_Specs"]["Snapshot_Frequency"]

    train_dir = os.path.join(base_dir, ds_specs["Training_Dir"])
    validation_dir = os.path.join(base_dir,ds_specs["Validation_Dir"])
    test_dir = os.path.join(base_dir,ds_specs["Testing_Dir"])
    
    transform = T_geometric.Compose([aug.rotate_perturbation_point_cloud_g,aug.shift_point_cloud_g,aug.random_scale_point_cloud_g,aug.minmax_norm_g])
    train_loader = get_pultree_dataloader(train_dir, batch_size, transform = transform)
    val_loader = get_pultree_dataloader(validation_dir, batch_size, transform = transform)
    test_loader = get_pultree_dataloader(test_dir, 1, shuffle = False, transform = aug.minmax_norm_g)

    if load_model:
        PGN.load_state_dict(torch.load(point_graph_network_path))
        print('Point-Graph-Network loaded.')
    PGN.point_graph_joint_training_mode()
    optimizer = optim.Adam(PGN.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_dice = 0
    test_model = None
    print('Point Graph Network Training:')
    for i in range(epoches):
        point_feedback, graph_feedback = PGN_run_epoch(train_loader, PGN, optimizer, criterion, device = device, is_train = True)
        train_point_acc, train_point_loss, train_point_dice = point_feedback 
        train_graph_acc, train_graph_loss, train_graph_dice = graph_feedback
    
        if i%Snapshot_Frequency==0:
            print(f'Training Epoch {i}, Point loss: {train_point_loss:.3f}, ', f' Point acc: {train_point_acc:.3f}', f' Point dice: {train_point_dice:.3f}', f'Graph loss: {train_graph_loss:.3f}, ', f' Graph acc: {train_graph_acc:.3f}', f' Graph dice: {train_graph_dice:.3f}')  
            
        if i%val_per_epoch==0 or i==epoches-1:
            point_feedback, graph_feedback = PGN_run_epoch(val_loader, PGN, optimizer, criterion, is_train = False, device = device)
            val_point_acc, val_point_loss, val_point_dice = point_feedback 
            val_graph_acc, val_graph_loss, val_graph_dice = graph_feedback
            print(f'Validation: , Point loss: {val_point_loss:.3f}, ', f' Point acc: {val_point_acc:.3f}', f' Point dice: {val_point_dice:.3f}', f'Graph loss: {val_graph_loss:.3f}, ', f' Graph acc: {val_graph_acc:.3f}', f' Graph dice: {val_graph_dice:.3f}')  
            
            if val_point_dice+val_graph_dice>best_val_dice:
                print('Update best validation model')
                best_val_dice = val_point_dice+val_graph_dice
                test_model = copy.deepcopy(PGN).to('cpu')
                if save_model:
                    print('saving Point Graph Network to ', point_graph_network_path)
                    torch.save(test_model.state_dict(), point_graph_network_path)
        if i%lr_decay_epoch==0 and i!=0: optimizer = reset_lr(optimizer, lr_ratio = lr_decay_ratio)

    if test_model==None:
        test_model = copy.deepcopy(PGN).to('cpu')
    point_feedback, graph_feedback = PGN_run_epoch(test_loader, test_model.to(device), optimizer, criterion, is_train = False, device = device)
    test_point_acc, test_point_loss, test_point_dice = point_feedback 
    test_graph_acc, test_graph_loss, test_graph_dice = graph_feedback
    print(f'Final testing, Point loss: {test_point_loss:.3f}, ', f' Point acc: {test_point_acc:.3f}', f' Point dice: {test_point_dice:.3f}', f'Graph loss: {test_graph_loss:.3f}, ', f' Graph acc: {test_graph_acc:.3f}', f' Graph dice: {test_graph_dice:.3f}')  
    if save_model:
        print('saving Point Graph Network to ', point_graph_network_path)
        torch.save(test_model.state_dict(), point_graph_network_path)

def IPGN_train_step(batch, model, optimizer, criterion, is_train=True, device = None, infer_bs = 2000):
    optimizer.zero_grad()  # Clear gradients.
    point_loss, total_pts, correct_pt, dice_avg = 0,0,0,0
    if is_train: model.train()
    else: model.eval()
    imp_pts = torch.stack([torch.from_numpy(pt[:infer_bs]) for pt in batch.implicit_points],dim=0).float().to(device)#[8, 6000, 3]
    imp_label = torch.stack([torch.from_numpy(l[:infer_bs]) for l in batch.implicit_labels],dim=0).long().to(device)
    
    label = torch.stack([torch.from_numpy(l) for l in batch.points_label],dim=0).long().to(device)
    ei = batch.edge_index.to(device)
    pts = torch.stack([torch.from_numpy(pt) for pt in batch.points],dim=0).float().to(device)#[8, 6000, 3]
    B,N,C = pts.shape
    nodes = batch.x.view(B,-1,C).float().to(device)#[8, 519, 3]

    if is_train:
        pt_outs = model(imp_pts, pts, nodes, ei)
        loss = criterion(pt_outs, imp_label)
        loss.backward()  # Derive gradients.
        optimizer.step()
    else:
        with torch.no_grad():
            pt_outs = model(imp_pts,pts,nodes, ei)
            loss = criterion(pt_outs, imp_label)

    l = loss.detach().item()

    with torch.no_grad():
        pred_use = torch.argmax(pt_outs, dim=1).detach().to('cpu')
        label_dice = imp_label.to('cpu')
        #print('pred_use',pred_use.shape)
        #print('label_dice',label_dice.shape)
        point_dice = macro_dice(pred_use,label_dice,19)
        correct_pts = pred_use.eq(imp_label.to('cpu')).sum().item()
    total_pts = imp_label.shape[0]*imp_label.shape[1]
    del loss, imp_label, ei, pts
    return l, correct_pts, total_pts, point_dice
def IPGN_run_epoch(loader, model, optimizer, criterion, is_train=True,device = None):
    total_pts = 0
    total_pts_correct = 0
    final_pt_loss = 0
    final_pt_dice = 0
    for batch_i, batch in enumerate(loader):
        pt_loss,pts_correct,pts_num, pt_dice = IPGN_train_step(batch, model, optimizer, criterion, is_train=is_train, device = device)
        total_pts_correct+=pts_correct
        total_pts+=pts_num
        final_pt_loss+=pt_loss*(len(batch.ptr)-1)
        final_pt_dice+=pt_dice*(len(batch.ptr)-1)
    pts_accuracy = total_pts_correct/total_pts
    final_pt_loss /= len(loader.dataset)
    final_pt_dice /= len(loader.dataset)
    return pts_accuracy, final_pt_loss, final_pt_dice
def implicit_network_train(specs, ds_specs, IPGN, device = None, base_dir = '/'):
    
    point_graph_network_path = os.path.join(base_dir, ds_specs["Model_Save_Path"], 'Point_Graph_Network')
    IPGN.point_graph_network.load_state_dict(torch.load(point_graph_network_path))
    print('Point-Graph-Network loaded')
    load_model = (specs["Implicit_Network_Training_Specs"]["load_IPGN"]==1)
    IPGN_path = os.path.join(base_dir, ds_specs["Model_Save_Path"], 'Implicit_Point_Graph_Network')
    if load_model:
        IPGN.load_state_dict(torch.load(IPGN_path))
        print('Implicit-Point-Graph-Network loaded')
    save_model = (specs["Implicit_Network_Training_Specs"]["save_IPGN"]==1)
    epoches = specs["Implicit_Network_Training_Specs"]["Num_Epoches"]
    batch_size = specs["Implicit_Network_Training_Specs"]["Batch_Size"]
    learning_rate = specs["Implicit_Network_Training_Specs"]["Learning_Rate"]
    lr_decay_ratio = specs["Implicit_Network_Training_Specs"]["Learning_Rate_Decay"]
    lr_decay_epoch = specs["Implicit_Network_Training_Specs"]["Lr_Decay_Epoches"]
    val_per_epoch = specs["Implicit_Network_Training_Specs"]["Validation_Per_Epoches"]
    Snapshot_Frequency = specs["Implicit_Network_Training_Specs"]["Snapshot_Frequency"]

    train_dir = os.path.join(base_dir, ds_specs["Training_Dir"])
    validation_dir = os.path.join(base_dir, ds_specs["Validation_Dir"])
    test_dir = os.path.join(base_dir, ds_specs["Testing_Dir"])
    transform = T_geometric.Compose([aug.rotate_perturbation_point_cloud_g,aug.shift_point_cloud_g,aug.random_scale_point_cloud_g,aug.minmax_norm_g])
    train_loader = get_pultree_dataloader(train_dir, batch_size, transform = transform)
    val_loader = get_pultree_dataloader(validation_dir, batch_size, transform = transform)
    test_loader = get_pultree_dataloader(test_dir, 1, shuffle = False, transform = aug.minmax_norm_g)

    IPGN.point_graph_network.implicit_training_mode()
    optimizer = optim.Adam(IPGN.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_dice = 0
    test_model = None
    print('Implicit Network Training:')
    for i in range(epoches):
        train_point_acc, train_point_loss, train_point_dice = IPGN_run_epoch(train_loader, IPGN, optimizer, criterion, device = device, is_train = True)
    
        if i%Snapshot_Frequency==0:
            print(f'Training Epoch {i}, Point loss: {train_point_loss:.3f}, ', f' Point acc: {train_point_acc:.3f}', f' Point dice: {train_point_dice:.3f}')  
            
        if i%val_per_epoch==0 or i==epoches-1:
            val_point_acc, val_point_loss, val_point_dice = IPGN_run_epoch(val_loader, IPGN, optimizer, criterion, is_train = False, device = device)
        
            print(f'Validation: , Point loss: {val_point_loss:.3f}, ', f' Point acc: {val_point_acc:.3f}', f' Point dice: {val_point_dice:.3f}')  
            
            if val_point_dice>best_val_dice:
                print('Update best validation model')
                best_val_dice = val_point_dice
                test_model = copy.deepcopy(IPGN).to('cpu')
                if save_model:
                    print('saving Implicit Point Graph Network to ', IPGN_path)
                    torch.save(test_model.state_dict(), IPGN_path)
        if i%lr_decay_epoch==0 and i!=0: optimizer = reset_lr(optimizer, lr_ratio = lr_decay_ratio)

    if test_model==None:
        test_model = copy.deepcopy(IPGN).to('cpu')
    test_point_acc, test_point_loss, test_point_dice = IPGN_run_epoch(test_loader, test_model.to(device), optimizer, criterion, is_train = False, device = device)#,full_resolution = True, full_test=full_voxel, save_volume = True, run_pulseg = run_pulseg,volume_save_dir = prediction_save_dir_component)
    print(f'Final testing, Point loss: {test_point_loss:.3f}, ', f' Point acc: {test_point_acc:.3f}', f' Point dice: {test_point_dice:.3f}')  
    if save_model:
        print('saving Implicit Point Graph Network to ', IPGN_path)
        torch.save(test_model.state_dict(), IPGN_path)
    