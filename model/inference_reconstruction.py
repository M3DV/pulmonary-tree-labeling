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
import csv


def IPGN_inference(batch, model, source_volume_dir = '/', device = None):
    model.eval()
    #Acquire volume
    volume_path = os.path.join(source_volume_dir, os.path.basename(batch.voxel_file[0]))
    print(volume_path)
    volume = np.load(volume_path)
    volume = volume[volume.files[0]]
    #Acquire all foreground voxels
    full_inference_pts = np.transpose(np.array(np.where(volume>0)),(1,0))
    full_inference_targets = volume[full_inference_pts[:,0],full_inference_pts[:,1],full_inference_pts[:,2]]
    maxs = np.amax(full_inference_pts,axis=0)
    mins = np.amin(full_inference_pts,axis=0)
    full_inference_pts_normed = torch.stack([torch.from_numpy((full_inference_pts-mins)/(maxs-mins))],dim=0)
    ei = batch.edge_index.to(device)
    pts = torch.stack([torch.from_numpy(pt) for pt in batch.points],dim=0).float().to(device)#[8, 6000, 3]
    B,N,C = pts.shape
    nodes = batch.x.view(B,-1,C).float().to(device)#[8, 519, 3]

    pt_predictions = model(full_inference_pts_normed, pts, nodes, ei, full_voxel = True)+1

    volume = np.zeros((batch.vol_size[0][0],batch.vol_size[0][1],batch.vol_size[0][2]))
    for ind in range(full_inference_pts.shape[0]):
        volume[full_inference_pts[ind,0],full_inference_pts[ind,1],full_inference_pts[ind,2]] = pt_predictions[ind]
    return volume, pt_predictions, full_inference_targets, os.path.basename(batch.voxel_file[0])

def implicit_inference_epoch(infer_loader, IPGN, source_volume_dir, device = None, save_volume = True, volume_save_dir = None):
    filenames, accuracies, dice_scores = [], [], []
    for batch_i, batch in enumerate(infer_loader):
        predicted_volume, predictions, targets, volume_filename = IPGN_inference(batch, IPGN, source_volume_dir = source_volume_dir, device = device)
        print(volume_filename)
        filenames.append(volume_filename)
        print('Accuracy:', (predictions==targets).sum().item()/len(targets))
        accuracies.append((predictions==targets).sum().item()/len(targets))
        point_dice = macro_dice(torch.from_numpy(predictions),torch.from_numpy(targets),19)
        print('Dice', point_dice)
        dice_scores.append(point_dice)
        if save_volume:
            ni_img = nib.Nifti1Image(predicted_volume, affine=np.eye(4))
            save_filename = os.path.join(volume_save_dir, volume_filename).replace('.npz','.nii.gz')
            print(save_filename)
            nib.save(ni_img, save_filename)
            print('saved')
        if batch_i ==3:
            break
    
    return filenames, accuracies, dice_scores
def dense_volume_reconstruction(specs, ds_specs, IPGN, device = None, base_dir = '/'):
    IPGN.point_graph_network.implicit_inference_mode()
    IPGN_path = os.path.join(base_dir, ds_specs["Model_Save_Path"], 'Implicit_Point_Graph_Network')

    source_volume_dir = os.path.join(base_dir, ds_specs["Volume_Source_Dir"])
    if_save_volume = (ds_specs["Save_Reconstruction_Volume"] == 1)
    save_volume_dir = os.path.join(base_dir, ds_specs["Result_Save_Path"])
    save_report = (ds_specs["Make_Report"] == 1)

    IPGN.load_state_dict(torch.load(IPGN_path))
    print('Implicit-Point-Graph-Network loaded')
    
    infer_dir = os.path.join(base_dir, ds_specs["Inference_Dir"])
    infer_loader = get_pultree_dataloader(infer_dir, 1, shuffle = False, transform = aug.minmax_norm_g)

    IPGN.point_graph_network.implicit_inference_mode()
    dice_scores = 0
    print('Running: Implicit Network inference for fast volume reconstruction:')
    
    filenames, accuracies, dice_scores = implicit_inference_epoch(infer_loader, IPGN, infer_dir, 
                                                                device = device, save_volume = if_save_volume, 
                                                                volume_save_dir = save_volume_dir)
    if save_report:
        report_path = os.path.join(save_volume_dir, ds_specs["Report_file_Name"])
        stats_report(filenames, accuracies, dice_scores, report_path)
    print(f' Point acc: {sum(accuracies)/len(accuracies):.3f}', f' Point dice: {sum(dice_scores)/len(dice_scores):.3f}')  
#todo
def stats_report(filenames, accuracies, dice_scores, save_path):
    fns = ['filename', 'accuracy', 'dice score']
    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fns)
        for i in range(len(filenames)):
            writer.writerow({'filename': filenames[i], 'accuracy': round(accuracies[i],3), 'dice score': round(dice_scores[i],3)})
