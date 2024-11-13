import torch
from torch_geometric.data import InMemoryDataset, Data
from torch_geometric.loader import DataLoader as pygeo_dataloader
import glob
import networkx as nx
import numpy as np
class PultreeDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.root = root
        self.data, self.slices = torch.load(self.processed_paths[0])
        
    @property
    def raw_file_names(self):
        return sorted(glob.glob(self.root+'/*.npz'))

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = []
        total_sample_points = 6000
        vol_files = self.raw_file_names
        g_files = [v_file.replace('data_npz','data_graph') for v_file in vol_files]
        g_files = [g_file.replace('.npz','.graphml') for g_file in g_files]
        if 'airway.npz' in vol_files[0]: node_pad = 519
        elif 'artery.npz' in vol_files[0]:node_pad = 1813        
        else: node_pad = 1691
        print('node_pad:', node_pad)
        #iterate over all correponding pairs of graph file and volume file
        for g_file,vol_file in zip(g_files,vol_files):
            g = nx.read_graphml(g_file)
            vol = np.load(vol_file)
            volume = vol[vol.files[0]]
            size = volume.shape
            all_edges,all_edge_cls,all_nodes,all_nodes_cls,all_edge_points = [],[],[],[],[]
            #fill in all edges by iterating all edges
            for start in g._adj:
                for endpoint in g._adj[start]:
                    #skip self loop
                    if start==endpoint:continue
                    #get current edge
                    curr = g._adj[start][endpoint]
                    #acquire all edges. e.g.: [1,43]
                    all_edges.append([int(start[1:]),int(endpoint[1:])])
                    
                    if 'coords_list' not in curr:
                        shortest_key = -1
                        #find shortest path among clique
                        if len(curr)!=1:
                            shortest_len = 2**31
                            for inn in curr:
                                edge_coords = curr[inn]['coords_list'].split(",")
                                if shortest_len>len(edge_coords):
                                    shortest_len = len(edge_coords)
                                    shortest_key = inn
                            curr['coords_list'] =  curr[shortest_key]['coords_list']
                        else:
                            curr['coords_list'] =  curr[0]['coords_list']
                    edge_coords = curr['coords_list'].split(",")
                    edge_coords = np.array([int(float(s)) for s in edge_coords]).reshape(-1,3)
                    classes=[]
                    for ec in edge_coords:
                        classes.append(volume[ec[2],ec[1],ec[0]])
                    counts = np.bincount(classes)
                    counts[0]=0
                    edge_cls = np.argmax(counts)
                    if edge_cls==0:
                        all_edge_cls.append(-1)
                    else:
                        all_edge_cls.append(edge_cls)
            #save all node's coordinates
            for zn,node in enumerate(g._node):
                node_coor = [int(float(g._node[node]['Z'])),int(float(g._node[node]['Y'])),int(float(g._node[node]['X']))]
                all_nodes.append(node_coor)
                rows = np.argwhere(zn == np.array(all_edges))
                involved_classes=[]
                #for all edges where the current node presents
                for row in rows:
                    es = all_edges[row[0]]
                    involved_classes.append(all_edge_cls[row[0]])
                involved_classes = np.unique(np.array(involved_classes))
                if len(involved_classes.tolist())>1:
                    if not 0 in involved_classes:
                        all_nodes_cls.append(-1)
                    elif len(involved_classes.tolist())>2:
                        all_nodes_cls.append(-1)
                    else:
                        all_nodes_cls.append(np.amax(involved_classes))
                else:
                    all_nodes_cls.append(np.unique(np.array(involved_classes)).tolist()[0])
            edge_to_remove = []
            for edge_ind, (edge_, edge_cls_, edge_pts) in enumerate(zip(all_edges, all_edge_cls,all_edge_points)):
                if edge_cls_== -1:
                    s,e = edge_[0],edge_[1]
                    s_rows = np.argwhere(s == np.array(all_edges))
                    e_rows = np.argwhere(e == np.array(all_edges))
                    s_rows = [sr[0] for sr in s_rows]
                    e_rows = [sr[0] for sr in e_rows]
                    #giving concreate class to distal edges
                    if (len(s_rows)==2 and len(e_rows)!=2) or (len(s_rows)!=2 and len(e_rows)==2):
                        if len(s_rows)==2:
                            alternatives = np.amax(np.unique(np.array(all_edge_cls)[e_rows]))
                        else:
                            alternatives = np.amax(np.unique(np.array(all_edge_cls)[e_rows]))
                        if alternatives!=-1:
                            all_edge_cls[edge_ind] = alternatives
                            distal_edge_relabel+=1
                           
                    #if both ends have other edges, and there is a common labeled 3rd node, making a clique, remove the clique(edge)
                    elif len(s_rows)>2 and len(e_rows)>2:
                        common = np.intersect1d(np.unique(np.array(all_edges)[s_rows,:]),np.unique(np.array(all_edges)[e_rows,:]))
                        if np.amax(np.array(all_nodes_cls)[common])!=-1:
                            edge_to_remove.append(edge_ind)
            #print(len(all_edges),len(all_edge_cls),len(all_edge_points))
            if len(edge_to_remove)>0:
                all_edges_tmp, all_edge_cls_tmp, all_edge_points_tmp = [],[],[]
                for i in range(len(all_edges)):
                    if i in edge_to_remove:continue
                    all_edges_tmp.append(all_edges[i])
                    all_edge_cls_tmp.append(all_edge_cls[i])
                    all_edge_points_tmp.append(all_edge_points[i])
                all_edges = all_edges_tmp
                all_edge_cls = all_edge_cls_tmp
                all_edge_points = all_edge_points_tmp
            all_nodes = torch.tensor(all_nodes)
            #create padded nodes feature and nodes label
            all_nodes_pad = torch.ones((node_pad,3))*(-10)
            all_nodes_pad[:all_nodes.shape[0]] = all_nodes
            all_nodes_cls = torch.tensor(all_nodes_cls)
            all_nodes_cls_pad = torch.ones((node_pad))*(-10)
            all_nodes_cls_pad[:all_nodes.shape[0]] = all_nodes_cls
            #finish
            all_edges = torch.tensor(all_edges)
            all_edges = torch.transpose(all_edges, 0, 1)
            all_edge_cls = torch.tensor(all_edge_cls)
            node_mask = torch.tensor(all_nodes_cls_pad!=-1)
            edge_mask = torch.tensor(all_edge_cls!=-1)

            points = np.transpose(np.nonzero(volume))#(318179, 3)
            targets = volume[points[:,0], points[:,1], points[:,2]]-1
            perm = np.random.permutation(points.shape[0])
            points = points[perm]
            targets = targets[perm]
            full_points = points[:total_sample_points]
            full_label = targets[:total_sample_points]

            implicit_points = points[total_sample_points:total_sample_points+10000]
            implicit_labels = targets[total_sample_points:total_sample_points+10000]

            d = Data(x = all_nodes_pad, edge_index = all_edges, y=all_nodes_cls_pad,y_edge = all_edge_cls, node_mask = node_mask, edge_mask = edge_mask, voxel_file = vol_file, points=full_points, points_label = full_label, implicit_points = implicit_points,implicit_points_ori = implicit_points, implicit_labels = implicit_labels, vol_size = size)
            data_list.append(d)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

def get_pultree_dataloader(dir, bs, transform=None, shuffle = True):
    dataset = PultreeDataset(dir, transform = transform)
    loader = pygeo_dataloader(dataset, batch_size=bs,shuffle=shuffle)
    return loader