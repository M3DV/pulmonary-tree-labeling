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

def graph_encoder_train_step(batch, model, optimizer, criterion, is_train=True, device = None,epoch=0,use_pointnet=False,compose=False):
    #print(batch)
    optimizer.zero_grad()  # Clear gradients.
    total_nodes,correct_nodes = 0,0
    if is_train:
        model.train()
    else:
        model.eval()
    ei = batch.edge_index.to(device)
    B,N,C = torch.stack([torch.from_numpy(pt) for pt in batch.points],dim=0).shape#[8, 6000, 3]
    nodes = batch.x.view(B,-1,C).float().to(device)#[8, 519, 3]
    y_use = ((batch.y[batch.node_mask])-1).detach().cpu()
    #print(y_use.shape)
    y_use_mask = (y_use!=-11)
    y_use = y_use[y_use_mask]
    #print(y_use.shape)
    y_use_onehot = torch.nn.functional.one_hot(y_use.long()).float().to(device)
    if is_train:
        node_out = model(None,nodes, ei)
        pred_use_onehot = (node_out[batch.node_mask])[y_use_mask]
        loss = criterion(pred_use_onehot, y_use_onehot.float())
    else:
        with torch.no_grad():
            node_out = model(None,nodes, ei)
            pred_use_onehot = (node_out[batch.node_mask])[y_use_mask]
            loss = criterion(pred_use_onehot, y_use_onehot.float())
    with torch.no_grad():
        pred_use_node = torch.argmax(pred_use_onehot, dim=1).detach().to('cpu')
        correct_nodes += pred_use_node.eq(y_use.to('cpu')).sum().item()
        total_nodes += ((y_use>-1000)*1).sum().item()
    if is_train:
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
    l = loss.detach().item()
    with torch.no_grad():
        dice = macro_dice(pred_use_node,y_use.to('cpu'),19)
    del loss,node_out
    return l, correct_nodes, total_nodes, dice
def point_encoder_train_step(batch, model, optimizer, criterion, is_train=True, device = None, epoch=0, use_pointnet=False, compose=False):
    #print(batch)
    optimizer.zero_grad()  # Clear gradients.
    total_pts,correct_pts = 0,0
    if is_train:
        model.train()
    else:
        model.eval()
    label = torch.stack([torch.from_numpy(l) for l in batch.points_label],dim=0).long().to(device)
    pts = torch.stack([torch.from_numpy(pt) for pt in batch.points],dim=0).float().to(device)#[8, 6000, 3]
    B,N,C = pts.shape
    if is_train:
        pt_out = model(pts,None, None)
        pt_loss = criterion(pt_out, label)#use this for ce loss only
    else:
        with torch.no_grad():
            pt_out = model(pts,None, None)
            pt_loss = criterion(pt_out, label)#use this for ce loss only
    with torch.no_grad():
        pred_use = torch.argmax(pt_out, dim=1).detach().to('cpu')
        correct_pts = pred_use.eq(label.to('cpu')).sum().item()
        total_pts = ((label!=-2)*1).sum().item()
    if is_train:
        pt_loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
    l = pt_loss.detach().item()
    with torch.no_grad():
        dice = macro_dice(pred_use,label.to('cpu'),19)
    del pt_loss,pt_out
    return l, correct_pts, total_pts, dice  
def pretrain_epoch(loader, model, optimizer, criterion, is_train=True,
                    device = None, use_pointnet=False,compose = False, 
                    point_or_graph = True):
    total_pts = 0
    total_pts_correct = 0
    total_loss = 0
    avg_dice = 0
    for batch_i, batch in enumerate(loader):
        if point_or_graph:
            loss, correct_pts, total_pt, dice = point_encoder_train_step(batch, model, optimizer, criterion, is_train=is_train, 
                                            device = device, use_pointnet = use_pointnet)
        else:
            loss, correct_pts, total_pt, dice = graph_encoder_train_step(batch, model, optimizer, criterion, is_train=is_train, 
                                            device = device, use_pointnet = use_pointnet)
        total_pts_correct+=correct_pts
        total_pts+=total_pt
        total_loss+=loss*(len(batch.ptr)-1)
        avg_dice+=dice*(len(batch.ptr)-1)
    accuracy_pts = total_pts_correct/total_pts
    total_loss /= len(loader.dataset)
    avg_dice /= len(loader.dataset)
    return total_loss,accuracy_pts,avg_dice
def encoder_networks_pretrain(specs, ds_specs, model, device = None, target = 'point', base_dir = '/'):

    point_or_graph = (target=='point')
    specs_current = specs["Point_Training_Specs"] if point_or_graph else specs["Graph_Training_Specs"]
    load_encoder = (specs_current["Load_Model"]==1)
    save_encoder = (specs_current["Save_Model"]==1)


    encoder_save_path = os.path.join(base_dir, ds_specs["Model_Save_Path"], specs_current["checkpoint_filename"]) 
    encoder_pred_head_path = os.path.join(base_dir, ds_specs["Model_Save_Path"], specs_current["prediction_head_checkpoint_filename"]) 

    epoches = specs_current["Num_Epoches"]
    batch_size = specs_current["Batch_Size"]
    learning_rate = specs_current["Learning_Rate"]
    lr_decay_ratio = specs_current["Learning_Rate_Decay"]
    lr_decay_epoch = specs_current["Lr_Decay_Epoches"]
    val_per_epoch = specs_current["Validation_Per_Epoches"]
    Snapshot_Frequency = specs_current["Snapshot_Frequency"]
    
    train_dir = os.path.join(base_dir, ds_specs["Training_Dir"])
    validation_dir = os.path.join(base_dir,ds_specs["Validation_Dir"])
    test_dir = os.path.join(base_dir,ds_specs["Testing_Dir"])

    transform = T_geometric.Compose([aug.rotate_perturbation_point_cloud_g,aug.shift_point_cloud_g,aug.random_scale_point_cloud_g,aug.minmax_norm_g])
    train_loader = get_pultree_dataloader(train_dir, batch_size, transform = transform)
    val_loader = get_pultree_dataloader(validation_dir, batch_size, transform = transform)
    test_loader = get_pultree_dataloader(test_dir, 1, shuffle = False, transform = aug.minmax_norm_g)
    
    if point_or_graph: 
        model.initial_point_pretrain_mode()
        if load_encoder:
            print('loading pretrain point encoder:',encoder_save_path)
            model.point_encoder.load_state_dict(torch.load(encoder_save_path))
            model.pt_cls_head.load_state_dict(torch.load(encoder_pred_head_path))
    else: 
        model.initial_graph_pretrain_mode()
        if load_encoder:
            print('loading pretrain graph encoder:',encoder_save_path)
            model.graph_encoder.load_state_dict(torch.load(encoder_save_path))
            model.node_cls_head.load_state_dict(torch.load(encoder_pred_head_path))

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    best_val_dice = 0
    test_model = None
    train_accuracies,validation_accuracies = [],[]
    print('Initial encoder pretraining:')
    for i in range(epoches):
        train_loss, train_acc, train_dice = pretrain_epoch(train_loader, model, optimizer, criterion, device = device, point_or_graph = point_or_graph)
        train_accuracies.append(train_acc)
        if i%Snapshot_Frequency==0:
            print(f'Training Epoch {i}, Total loss: {train_loss:.3f}, ', f' accuracy: {train_acc:.3f}', f' dice score: {train_dice:.3f}')  
        if i%val_per_epoch==0 or i==epoches-1:
            val_loss, val_acc, val_dice = pretrain_epoch(val_loader, model, optimizer, criterion, is_train = False, device = device,point_or_graph = point_or_graph)
            validation_accuracies.append(val_acc)
            print(f'Validation loss: {val_loss:.3f}, ', f' accuracy: {val_acc:.3f}', f' dice score: {val_dice:.3f}')
            if val_dice>best_val_dice:
                print('Update best validation model')
                best_val_dice = val_dice
                test_model = copy.deepcopy(model).to('cpu')
                if save_encoder:
                    print('saving encoder:', encoder_save_path)
                    if point_or_graph:
                        torch.save(test_model.point_encoder.state_dict(), encoder_save_path)
                        torch.save(test_model.pt_cls_head.state_dict(), encoder_pred_head_path)
                    else:
                        torch.save(test_model.graph_encoder.state_dict(), encoder_save_path)
                        torch.save(test_model.node_cls_head.state_dict(), encoder_pred_head_path)
        if i%lr_decay_epoch==0 and i!=0: optimizer = reset_lr(optimizer,lr_ratio = lr_decay_ratio)

    if test_model==None:
        test_model = copy.deepcopy(model).to('cpu')
    test_loss, test_acc_node, test_dice = pretrain_epoch(test_loader, test_model.to(device),optimizer, 
                                        criterion, is_train = False, device = device, point_or_graph = point_or_graph)
    print(f'Final test loss: {test_loss:.3f}', f'Final acc_nodes: {test_acc_node:.3f}', f'Final dice_nodes: {test_dice:.3f}')
    if save_encoder:
        print('saving encoder model to ', encoder_save_path)
        if point_or_graph:
            torch.save(test_model.point_encoder.state_dict(), encoder_save_path)
            torch.save(test_model.pt_cls_head.state_dict(), encoder_pred_head_path)
        else:
            torch.save(test_model.graph_encoder.state_dict(), encoder_save_path)
            torch.save(test_model.node_cls_head.state_dict(), encoder_pred_head_path)



