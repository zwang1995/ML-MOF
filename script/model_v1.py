import torch
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
from torch import nn
from torch.nn import GRU
from torch_geometric.nn import Set2Set


class MOFNet(nn.Module):
    def __init__(self, params, ni, dim, aggr, act=None):
        super(MOFNet, self).__init__()
        self.params = params

        if act == "tanh":
            self.act = torch.tanh
        elif act == "sigmoid":
            self.act = torch.sigmoid
        elif act == "relu":
            self.act = torch.relu
        elif act == "softplus":
            self.act = F.softplus
        elif act == "elu":
            self.act = F.elu

        if not self.params["use_n2v_emb"]:
            self.iden_emb = nn.Embedding(params["iden_num"], 8)
            self.linear0 = nn.Linear(8, dim)
        else:
            self.iden_emb = nn.Embedding(params["node_num"], params["n2v_dim"])
            self.linear0 = nn.Linear(params["n2v_dim"], dim)

        # 1a. Organic linker
        def get_conv(ni):
            conv0 = pyg_nn.GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
            conv1 = pyg_nn.GCNConv(dim, dim, aggr=aggr, add_self_loops=True)
            conv2 = pyg_nn.AGNNConv(add_self_loops=True)
            # Attention-based Graph Neural Network 2
            conv3 = pyg_nn.ClusterGCNConv(dim, dim, aggr=aggr, add_self_loops=True)
            # Cluster-GCN: An Efficient Algorithm for Training Deep and Large Graph Convolutional Networks 2
            conv4 = pyg_nn.GATConv(dim, dim, aggr=aggr, add_self_loops=True)
            # Graph Attention Networks 2
            conv5 = pyg_nn.GraphConv(dim, dim, aggr=aggr)
            # Weisfeiler and Leman Go Neural: Higher-order Graph Neural Networks 2
            conv6 = pyg_nn.LEConv(dim, dim, aggr=aggr)
            # ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical Graph Representations 2
            conv7 = pyg_nn.MFConv(dim, dim, aggr=aggr)
            # Convolutional Networks on Graphs for Learning Molecular Fingerprints 2
            conv8 = pyg_nn.SAGEConv(dim, dim, aggr=aggr)
            # Inductive Representation Learning on Large Graphs 2
            convs = [conv0, conv1, conv2, conv3, conv4, conv5, conv6, conv7, conv8]
            conv = convs[ni]
            return conv

        self.conv = get_conv(ni)
        self.gru = GRU(dim, dim)
        self.set2set = Set2Set(dim, processing_steps=2)

        # 1b. Metal node
        self.node_emb = nn.Embedding(params["node_num"], params["embed_dim"])

        # 1c. Topology
        self.topo_emb = nn.Embedding(params["topo_num"], params["embed_dim"], padding_idx=params["topo_pad"])

        # 2. Feed-forward
        dim_ = len(params["struc"]) if params["use_struc"] else 0
        dim__ = 1 if params["use_pres"] else 0
        fc_dim = 2 * dim + dim_ + dim__ + 2 * params["embed_dim"]

        self.fc1 = torch.nn.Linear(fc_dim, 2 * dim)
        self.fc11 = torch.nn.Linear(2 * dim, dim)
        self.fc12 = torch.nn.Linear(dim, dim)
        self.fc13 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        if not self.params["use_n2v_emb"]:
            x = self.iden_emb(data.x)
        else:
            x = data.x

        out = self.act(self.linear0(x))
        h = out.unsqueeze(0)
        for i in range(3):
            m = self.act(self.conv(out, data.edge_index)) + out
            out, h = self.gru(m.unsqueeze(0), h)
            out = out.squeeze(0)
        out = self.set2set(out, data.batch)

        out_node = self.node_emb(data.node)
        out_topo = self.topo_emb(data.topo)

        if self.params["use_struc"]:
            if self.params["use_pres"]:
                out = torch.cat([out, out_node, out_topo, torch.tensor(data.struc, dtype=torch.float32),
                                 torch.unsqueeze(torch.tensor(data.p, dtype=torch.float32), 1)], dim=1)
            else:
                out = torch.cat([out, out_node, out_topo, torch.tensor(data.struc, dtype=torch.float32)], dim=1)
        else:
            out = torch.cat([out, out_node, out_topo], dim=1)

        x1 = self.act(self.fc1(out))
        x1 = self.act(self.fc11(x1))
        x1 = self.act(self.fc12(x1))
        x1 = self.fc13(x1)

        return x1
