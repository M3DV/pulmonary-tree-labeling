import torch
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn.norm import GraphNorm

class GAT(torch.nn.Module):
    def __init__(self, hidden_channels, heads=2, input_channel = 3, out=False):
        super().__init__()
        torch.manual_seed(1234567)
        all_channels = [input_channel]+hidden_channels
        self.node_forward = torch.nn.ModuleList([])
        for i in range(len(all_channels)-1):
            self.node_forward.append( GATv2Conv(all_channels[i], all_channels[i+1],heads = heads,concat=False))
            if i<len(all_channels)-2:
                self.node_forward.append(GraphNorm(in_channels = all_channels[i+1]))
                self.node_forward.append(torch.nn.ReLU(inplace=True))
            elif not out:
                self.node_forward.append(GraphNorm(in_channels = all_channels[i+1]))
                self.node_forward.append(torch.nn.ReLU(inplace=True))
    def forward(self, x, edge_index, is_train=True):
        for ind,m in enumerate(self.node_forward):
            if isinstance(m, GATv2Conv):
                x = m(x,edge_index)
            else:
                x = m(x)
        #src, dst = edge_index
        #edge_fea = (x[src] + x[dst])/2
        #for e_l in self.mlp:
            #x = e_l(x)
            #edge_fea = e_l(edge_fea)
        return x
class GAT_mlp(torch.nn.Module):
    def __init__(self, mlp_channels, input_channel = 512, last_layer = True):
        super().__init__()
        torch.manual_seed(1234567)
        mlp_channels = [input_channel]+mlp_channels
        if last_layer:mlp_channels = mlp_channels+[64,19]
        self.mlp = torch.nn.ModuleList([])
        for i in range(len(mlp_channels)-2):
            self.mlp.append(torch.nn.Linear(mlp_channels[i], mlp_channels[i+1]))
            self.mlp.append(torch.nn.ReLU(inplace=True))
            #self.mlp.append(nn.Dropout(p=0.2))
        self.mlp.append(torch.nn.Linear(mlp_channels[-2], mlp_channels[-1]))
    def forward(self, x, is_train=True):
        for ind,m in enumerate(self.mlp):
            x = m(x)
        return x