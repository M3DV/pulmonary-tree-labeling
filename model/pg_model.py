import torch.nn as nn
from model.pointnet2 import Pointnet2
import model.GAT as GAT
import model.point_graph_func as pg_func 
import torch.nn.functional as F
from model.utilities import *
import time
import numpy as np

class IPGN(nn.Module):
    def __init__(self, specs, num_class = 19, max_node=519, device = 'cpu'):
        super(IPGN, self).__init__()
        """
        Initialize the Implicit Point-Graph Network. 
    
        Args:
        specs: A Json object that contains critical configuration details for the network. 
        
        Variables:
        self.point_graph_network: The Point-Graph Network
        self.feature_input: a list of integers as indices, representing the point-graph fusion layers
                            from which feature will be extracted for usage.
        self.PN_FeatProp: A feature propagation module (Figure 8 (b-c) in manuscript)        
        self.pt_cls_head: Final classification layers module (Figure 8 (c) in manuscript)       
        """
        self.point_graph_network = point_graph_network(specs, num_class = num_class, max_node = max_node, device = device, stage = 0)
        initial_dim = specs["Point_Graph_Fusion"]["initial_feature_dim"]
        block_out_dim = specs["Point_Graph_Fusion"]["blks_dim"]
        feature_input = specs["Implicit_Module"]["pgn_layer_imp_input"]
        fp_input = sum([block_out_dim[i] for i in feature_input])+initial_dim
        self.feature_input=feature_input
        fp_tmp = fp_input
        fp_hidden = []
        while fp_tmp>=128:
            fp_hidden.append(int(fp_tmp))
            fp_tmp/=2
        self.PN_FeatProp = pg_func.PointNetFeaturePropagation(fp_input,fp_hidden+[128],fp_num=3)#or 'concat'                    
        self.pt_cls_head = pg_func.classification_head(128,[128,64],num_class)
        self.device = device

    def forward(self, infer_points, points, nodes, ei, full_voxel=False, infer_n = 6000, bs=4):
        """
        Forward pass for the Implicit Point-Graph Network. 
        In full_voxel mode, inference points are the dense points in a 3D image for final inference for reconstruction.
        In not full_voxel mode, inference points are randomly sampled sparse points for training purpose.  
        
        Args:
        infer_points (torch.Tensor): Tensor (B, N, 3) representing input query points, 
                            where B is the batch size, N is the number of query points, and 3 is the 3D coordinates.
        points (torch.Tensor): Tensor (B, N, 3) representing input point cloud, 
                            where B is the batch size, N is the number of points, and 3 is the 3D coordinates.
        nodes (torch.Tensor): Tensor (B, S, 3) representing input node features, 
                              where B is the batch size, S is the number of nodes, and 3 is the 3D coordinates.
        ei (torch.Tensor): Edge index or connectivity information for graph-based processing.
        full_voxel (bool, optional): Flag indicating whether the forward operation is for dense reconstruction.
        infer_n (int): infer_points are broken into batches with "infer_n" number of points for parallel inference.
        bs (int): Batch size for each batched inference operations for the infer_points.

        Returns:
        full_prediction (list): A list of multi-class segmentation prediction ranging from 0 to class number-1.(full_voxel == True) 
        p_out (torch.Tensor): Tensor (B, C, N) representing predicted class logits.(full_voxel == False) 
        """
        points_feature, p_outs,n_outs,p_features,n_features = self.point_graph_network(points, nodes, ei)
        p_f = torch.cat([p_features[i] for i in self.feature_input],2)
        p_f = torch.cat((p_f,points_feature),2)  
        last_bit = False

        if full_voxel:
            ind=0
            #infer_points have batch size = 1 when performing full voxel reconstruction.
            infer_points =  infer_points[0] 
            total_n = infer_points.shape[0]
            full_prediction = np.zeros(total_n)
            #create batched inference operations for accelerated reconstruction  
            while ind+bs*infer_n<total_n:
                start = ind
                ind+=bs*infer_n
                imp_pts = infer_points[start:ind].reshape(bs,infer_n,3).float().to(self.device)
                points_tmp = torch.cat([points for i in range(bs)],0)
                p_f_tmp = torch.cat([p_f for i in range(bs)],0)
                p_fea = self.PN_FeatProp(imp_pts.permute(0, 2, 1),points_tmp.permute(0, 2, 1),None,p_f_tmp.permute(0, 2, 1))
                p_out = self.pt_cls_head(p_fea)
                p_pred = torch.argmax(p_out, dim=1).to('cpu').reshape(-1).numpy()
                full_prediction[start:ind] =  p_pred
            imp_pts = infer_points[ind:].float().reshape(1,-1,3).to(self.device)
            p_fea = self.PN_FeatProp(imp_pts.permute(0, 2, 1),points.permute(0, 2, 1),None,p_f.permute(0, 2, 1))
            p_out = self.pt_cls_head(p_fea)
            p_pred = torch.argmax(p_out, dim=1).to('cpu').reshape(-1).numpy()
            full_prediction[ind:] =  p_pred
            return full_prediction
        else:
            p_fea = self.PN_FeatProp(infer_points.permute(0, 2, 1),points.permute(0, 2, 1),None,p_f.permute(0, 2, 1))
            return self.pt_cls_head(p_fea)

class point_graph_network(nn.Module):
    def __init__(self, specs, num_class = 19, max_node=519, device = 'cpu', stage = 0):
        super(point_graph_network, self).__init__()
        """
        Initialize the Point-Graph Network, which takes both point-based and graph-based input, 
        and produce corresponding multi-class prediction. 
        This module can also run only the initial encoders with stage = 0/1, for pretraining purpose.
    
        Args:
        specs: A Json object that contains critical configuration details for the network. 
    
        Variables:
        self.stage: Different training stages.
                    0 - point initial pretrain.
                    1 - graph initial pretrain.
                    2 - point-graph fusion.
        self.point_encoder: Initial Point-based encoder (Pointnet++).
        self.graph_encoder: Initial Graph-based encoder (GAT).
        self.blocks: the point-graph fusion layer module.
        self.pt_cls_head: the auxiliary prediction head for initial point-based encoder.
        self.node_cls_head: the auxiliary prediction head for initial graph-based encoder.

        Returns:
        Depending on the training stage:
            Stage 0 - p_out: Tensor of point classification outputs with shape (B, Num_class = 19, N)
            Stage 1 - n_out: Tensor of node classification outputs with shape (Num_Node, Num_class = 19)
            Stage 2&3 - points_feature: Feature from point encoder with shape (B, Num_pts, feature_dimension = 128)
                        p_outs: List of point predictions from all point-graph fusion layers
                                [(B, Num_class, Num_pts), ..., (B, Num_class, Num_pts)]
                        n_outs: List of node (graph) predictions from all point-graph fusion layers
                                [(Num_nodes, Num_class), ..., (Num_nodes, Num_class)]
                        p_features: List of point features out of point-graph fusion layers in point branch
                                [(B, Num_pts, Feature_Dim), ..., (B, Num_pts, Feature_Dim)]
                        n_features: List of node features out of point-graph fusion layers in graph branch
                                [(B, Max_Num_Node, Feature_Dim), ..., (B, Max_Num_Node, Feature_Dim)]
        """
        initial_dim = specs["Point_Graph_Fusion"]["initial_feature_dim"]
        block_out_dim = specs["Point_Graph_Fusion"]["blks_dim"]
        radius = specs["Point_Graph_Fusion"]["radius"]
        nsample = specs["Point_Graph_Fusion"]["nsample"]
        fp_nums = specs["Point_Graph_Fusion"]["fp_nums"]
        p_op, n_op = specs["Point_Graph_Fusion"]["p_operation"], specs["Point_Graph_Fusion"]["n_operation"]
        self.num_class = num_class
        
        self.point_encoder = Pointnet2(specs["Point_Encoder"])

        gat_hidden = specs["Graph_Encoder"]["GNN_Hidden_Layers"]
        gat_attention_head = specs["Graph_Encoder"]["Attention_Head_Num"]
        gat_input_channel = specs["Graph_Encoder"]["Input_Channel"]
        self.graph_encoder = GAT.GAT(gat_hidden, heads = gat_attention_head, input_channel = gat_input_channel, out=False)

        last_channel = initial_dim
        self.blocks = torch.nn.ModuleList([])
        self.stage = stage
        last = False

        for ind,dim in enumerate(block_out_dim):
            if ind==len(block_out_dim)-1:last = True
            blk = pg_fusion_layer(radius[ind],nsample,last_channel,dim, device = device,
                                  max_node = max_node, fp_num = fp_nums[ind], p_operation = p_op, n_operation = n_op)
            self.blocks.append(blk)
            last_channel = dim
        
        #pt_cls_head and node_cls_head are auxilary neural networks that enable the initial point and graph encoders to make predictions.
        self.pt_cls_head = pg_func.classification_head(last_channel,[128,64],num_class)
        self.node_cls_head = GAT.GAT([128,64,num_class], heads = 2, input_channel = last_channel,out=True)

    def forward(self, points, nodes, ei,is_train = True):
        """
        Forward pass for the Point-Graph Network, which also supports forward pass for the point-based,
        or graph-based encoders only, determined by the 'stage' parameter.
    
        Args:
        points (torch.Tensor): Tensor (B, N, 3) representing input point cloud, 
                            where B is the batch size, N is the number of points, and 3 is the 3D coordinates.
        nodes (torch.Tensor): Tensor (B, S, 3) representing input node features, 
                              where B is the batch size, S is the number of nodes, and 3 is the 3D coordinates.
        ei (torch.Tensor): Edge index or connectivity information for graph-based processing.
        is_train (bool, optional): Flag indicating whether the model is in training mode. Default True.
    
        Returns:
        Depending on the training stage:
            Stage 0 - p_out (torch.Tensor): Tensor of point classification outputs with shape (B, Num_class, N)
            Stage 1 - n_out (torch.Tensor): Tensor of node classification outputs with shape (Num_Node, Num_class)
            Stage 2&3 - points_feature (torch.Tensor): Feature from point encoder with shape (B, Num_pts, dim)
                        p_outs: List of point predictions from all point-graph fusion layers
                                [(B, Num_class, Num_pts), ..., (B, Num_class, Num_pts)]
                        n_outs: List of node (graph) predictions from all point-graph fusion layers
                                [(Num_nodes, Num_class), ..., (Num_nodes, Num_class)]
                        p_features: List of point features out of point-graph fusion layers in point branch
                                [(B, Num_pts, Feature_Dim), ..., (B, Num_pts, Feature_Dim)]
                        n_features: List of node features out of point-graph fusion layers in graph branch
                                [(B, Max_Num_Node, Feature_Dim), ..., (B, Max_Num_Node, Feature_Dim)]
        """
        points_feature, gnn_feature = None, None
        if self.stage == 0:
            B,N,C = points.shape
            points_feature,_ = self.point_encoder(points.permute((0,2,1)))
            p_out = self.pt_cls_head(points_feature.permute((0,2,1)))
            return p_out
        elif self.stage == 1:
            B,S,C = nodes.shape
            gnn_feature = self.graph_encoder(nodes.reshape(-1,3),ei)#.reshape(B,S,-1) 
            n_out = self.node_cls_head(gnn_feature, ei)
            return n_out
        else:
            B,N,C = points.shape
            _,S,_ = nodes.shape
            points_feature,_ = self.point_encoder(points.permute((0,2,1)))
            gnn_feature = self.graph_encoder(nodes.reshape(-1,3),ei).reshape(B,S,-1) 
            p_outs,n_outs,p_features,n_features = [],[],[],[]
            for ind,blk in enumerate(self.blocks):
                points_feature,gnn_feature,p_out,n_out = blk(points_feature, points,gnn_feature, nodes, ei )
                p_outs.append(p_out)
                n_outs.append(n_out)
                p_features.append(points_feature)
                n_features.append(gnn_feature)
            gnn_out_dim = gnn_feature.shape[2]
            gnn_feature = gnn_feature.reshape(-1,gnn_out_dim)
            return points_feature,p_outs,n_outs,p_features,n_features

    #Set PGN to only training the point-encoder (Pointnet++) parameters. 
    def initial_point_pretrain_mode(self):
        self.stage = 0
        for param in self.parameters():
            param.requires_grad = False
        for param in self.point_encoder.parameters():
            param.requires_grad = True
        for param in self.pt_cls_head.parameters():
            param.requires_grad = True

    #Set PGN to only training the graph-encoder (GAT Network) parameters. 
    def initial_graph_pretrain_mode(self):
        self.stage = 1
        for param in self.parameters():
            param.requires_grad = False
        for param in self.graph_encoder.parameters():
            param.requires_grad = True
        for param in self.node_cls_head.parameters():
            param.requires_grad = True

    #Set PGN to training the Point-Graph fusion layer parameters. 
    def point_graph_joint_training_mode(self):
        self.stage = 2
        for param in self.parameters():
            param.requires_grad = True
        for param in self.point_encoder.parameters():
            param.requires_grad = False
        for param in self.pt_cls_head.parameters():
            param.requires_grad = False
        for param in self.graph_encoder.parameters():
            param.requires_grad = False
        for param in self.node_cls_head.parameters():
            param.requires_grad = False
    
    def implicit_training_mode(self):
        #Set PGN to training the Point-Graph fusion layer parameters. 
        self.stage = 2
        for param in self.parameters():
            param.requires_grad = False

    def implicit_inference_mode(self):
        #Set PGN to inference the Point-Graph fusion layer parameters. 
        self.stage = 3
        for param in self.parameters():
            param.requires_grad = False

class pg_fusion_layer(nn.Module):
    def __init__(self,bq_radius,nsample,in_channel,out_channel, max_node = 519,device = 'cpu',last = False,
                        fp_num=3,p_operation = 0,n_operation=1,num_class=19):
        super(pg_fusion_layer, self).__init__()
        """
        Initialize the Point-Graph Fusion Layer, which takes input feature from both the point-based and graph-based branch, 
        and produce corresponding point-based and graph-based updated feature.
    
        Args:
        bq_radius: The radius of the ball query operation, could range from [0,1], and defaulted to be 0.1.
        nsample: Max sample number in local ball region.
        in_channel: The dimension of the input feature for both point and graph branch.
        out_channel: The dimension of the output feature for both point and graph branch.
        fp_num: The number of neighbors to search for in the K-nn process. 
        p_operation: The type of operation for point to search for neighboring node for feature fusion. (* default in paper)
                     0 = Ball query 
                     1 = K-NN (*) 
        n_operation: The type of operation for node to search for neighboring points for feature fusion. (* default in paper)
                     0 = Ball query (*) 
                     1 = K-NN 
        self.PN_FeatProp: Feature propagation module for point operation (p_operation=1, Figure 7, F2 in manuscript).
        self.group_layer: A module that unite feature of multiple queried points to one (n_operation=0, Figure 7, F1 in manuscript).
        self.GNN: A lightweight GNN to update graph node feature (n_operation=0, Figure 7, G in manuscript).
        self.pt_cls_head: A prediction head for point-based predictions for deep supervision.
        self.node_cls_head: A GNN prediction head for graph-based predictions for deep supervision.
        num_class: number of class.
        """

        self.device = device
        self.point_operation,self.node_operation = p_operation, n_operation
        self.bq_radius = bq_radius
        self.nsample = nsample
        self.in_channel = in_channel
        self.out_channel = out_channel
        p_mlp = [out_channel for i in range(3)]
        self.PN_FeatProp = pg_func.PointNetFeaturePropagation(in_channel*2,p_mlp,fp_num=fp_num)#or 'concat'
        self.group_layer = pg_func.Grouping_MLP_MP(in_channel,[in_channel for i in range(3)])
        gnn_hidden = [out_channel for i in range(3)]
        if last: gnn_hidden = gnn_hidden+[64,num_class]
        self.GNN = GAT.GAT(gnn_hidden, input_channel=in_channel*2)
        #for when point feature is from ball query and grouping
        if self.point_operation==0:
            self.pts_merge_mlp = pg_func.PGMLP(in_channel*2, [in_channel]+[out_channel for i in range(2)])
        if self.node_operation==1:
            self.GNN_FeatProp = pg_func.PointNetFeaturePropagation(in_channel*2,p_mlp,fp_num=fp_num)#or 'concat'
        self.pt_cls_head = pg_func.classification_head(out_channel,[128,64],num_class)
        self.node_cls_head = GAT.GAT([128,64,num_class], heads = 2, input_channel = out_channel,out=True)
        self.max_node = max_node
        self.is_last = last
    def forward(self,points_fea, points,nodes_fea, nodes, ei,is_train = True):
        '''
        Args:
        points (torch.Tensor): Tensor (B, N, 3) representing input points, 
        points_fea (torch.Tensor): Tensor (B, N, D) representing input point features, 
        nodes (torch.Tensor): Tensor (B, S, 3) representing input nodes, 
        nodes_fea (torch.Tensor): Tensor (B, S, D) representing input node features, 
        ei (torch.Tensor): Edge index or connectivity information for graph-based processing.
        is_train (bool, optional): Flag indicating whether the model is in training mode. Default True.

        Returns:
        pt_new_feature (torch.Tensor): Tensor (B, N, D1) representing updated point-based feature with dimension D1. 
        nodes_new_feature (torch.Tensor): Tensor (B, S, D2) representing updated point-based feature with dimension D2. 
        p_out (torch.Tensor): Tensor (B, Num_class, N) representing the point-based class prediction with new feature.
        n_out (torch.Tensor): Tensor (N, Num_class) representing the graph-based class prediction with new feature.
        '''
        src,dst = ei
        B,N,C = points.shape
        B,N,D = points_fea.shape
        _,S,_ = nodes.shape
        _,S,D = nodes_fea.shape

        #For Node:
        #perform ball query+grouping for node feature update
        if self.node_operation==0:
            #here
            #ball query from Nodes on all points
            group_idx = query_ball_point_custom(self.bq_radius, self.nsample, points, nodes) #[B, S, nsample]
            #using raw coordinates feature
            grouped_feature = index_points(points_fea, group_idx)# [B, S, nsample, 3])
            nodes_new_feature = self.group_layer(grouped_feature)#4,519,128
            nodes_new_feature = nodes_new_feature.reshape(-1,self.in_channel)
        #perform feature propagation for node feature
        else:
            #print(nodes.permute((0,2,1)).shape,points.permute((0,2,1)).shape,nodes_fea.permute(0,2,1).shape,points_fea.permute(0,2,1).shape)
            nodes_new_feature = self.GNN_FeatProp(nodes.permute((0,2,1)),points.permute((0,2,1)),nodes_fea.permute(0,2,1),points_fea.permute(0,2,1))
            nodes_new_feature = nodes_new_feature.permute((0,2,1)).reshape(B*S,-1)
        nodes_fea = nodes_fea.reshape(-1,self.in_channel)
        nodes_new_feature = torch.cat((nodes_new_feature,nodes_fea),1)
        #Perform Graph learning
        nodes_new_feature = self.GNN(nodes_new_feature,ei) #[B*S, C_3]
        if not self.is_last:nodes_new_feature=nodes_new_feature.reshape(B,S,self.out_channel)
        else: nodes_new_feature=nodes_new_feature.reshape(B,S,19)
        #print('Node feature after GNN learning:', gnn_node_fea.shape)

        if self.point_operation==0:
            group_idx = query_ball_point_custom(self.bq_radius, self.nsample, nodes, points) #[B, S, nsample]
            #using raw coordinates feature
            grouped_feature = index_points(nodes_fea.reshape(B,S,-1), group_idx)
            pt_new_feature = self.group_layer(grouped_feature)
            pt_new_feature = torch.cat((pt_new_feature,points_fea),2)
            pt_new_feature = self.pts_merge_mlp(pt_new_feature.permute((0,2,1)))
            pt_new_feature = pt_new_feature.permute((0,2,1))
        #perform ball query+grouping for point feature
        else:
            nodes_fea = nodes_fea.reshape(B,S,D).permute((0,2,1))
            pt_new_feature = self.PN_FeatProp(points.permute(0,2,1),nodes.permute(0,2,1),points_fea.permute(0,2,1),nodes_fea)
        n_out = self.node_cls_head(nodes_new_feature.reshape(-1,self.out_channel),ei)
        p_out = self.pt_cls_head(pt_new_feature)
        return pt_new_feature.permute((0,2,1)), nodes_new_feature, p_out, n_out
