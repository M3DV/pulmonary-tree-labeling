import os
import json
import torch
import numpy as np
from model.utilities import *
import model.data_augmentation as aug
from torch_geometric.loader import DataLoader as pygeo_dataloader

from model.pygeo_dataset import PultreeDataset, get_pultree_dataloader
import model.point_graph_func as pg_function
import model.pg_model as pg_model
import torch_geometric.transforms as T_geometric
from torch import optim
from model.encoder_pretrain import encoder_networks_pretrain
from model.pg_fusion_training import point_graph_network_train, implicit_network_train
from model.inference_reconstruction import dense_volume_reconstruction
import copy
import gc
import argparse


def train():
    dataset_file = parse_arg()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        print("Training on GPU")
        device = torch.device("cuda:1")
    print(device)

    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    print('Project directory:', BASE_DIR)

    ds_config_file = os.path.join(BASE_DIR, 'specs',dataset_file)
    with open(ds_config_file) as dataset_specs_file:
        dataset_specs = dataset_specs_file.read()
        dataset_specs = json.loads(dataset_specs)
    dataset_name = dataset_specs["Dataset_Name"]
    max_node = dataset_specs["Max_Node"]
    num_class = dataset_specs["Num_Class"]

    network_config_file = os.path.join(BASE_DIR, 'specs','network_specs.json')
    with open(network_config_file) as network_specs_file:
        network_specs = network_specs_file.read()
        network_specs = json.loads(network_specs)

    train_config_file = os.path.join(BASE_DIR, 'specs','train_inference_specs.json')
    with open(train_config_file) as train_specs_file:
        train_inf_specs = train_specs_file.read()
        train_inf_specs = json.loads(train_inf_specs)

    print('Currently processing:', dataset_name)
    IPGN = pg_model.IPGN(network_specs, num_class = num_class, max_node = max_node, device = device).to(device)
    PGN = IPGN.point_graph_network
    encoder_networks_pretrain(train_inf_specs, dataset_specs, PGN, device = device, target = 'point', base_dir = BASE_DIR) #Training/testing of the initial point-based encoder (Pointnet++)
    encoder_networks_pretrain(train_inf_specs, dataset_specs, PGN, device = device, target = 'graph', base_dir = BASE_DIR) #Training/testing of the initial graph-based encoder (GAT)
    point_graph_network_train(train_inf_specs, dataset_specs, PGN, device = device, base_dir = BASE_DIR) #Training/testing of the Point-Graph-Network
    implicit_network_train(train_inf_specs, dataset_specs, IPGN, device = device, base_dir = BASE_DIR) #

def parse_arg():
    parser = argparse.ArgumentParser(prog='ProgramName')
    parser.add_argument('-d','--dataset')   
    args = parser.parse_args()
    return args.dataset

if __name__ == '__main__':
    train()