import torch
import torch.nn.functional as F
from torch import nn


class MOFNet_FNN(nn.Module):
    def __init__(self, params, ni, dim, aggr, act=None):
        super(MOFNet_FNN, self).__init__()
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

        self.topo_emb = nn.Embedding(params["topo_num"], params["embed_dim"], padding_idx=params["topo_pad"])

        self.fc1 = torch.nn.Linear(len(self.params["struc"]) + params["embed_dim"], 2 * dim)
        self.fc11 = torch.nn.Linear(2 * dim, dim)
        self.fc12 = torch.nn.Linear(dim, dim)
        self.fc13 = torch.nn.Linear(dim, 1)

    def forward(self, data):
        out_topo = self.topo_emb(data.topo)
        if self.params["use_struc"]:
            out = torch.cat([out_topo, torch.tensor(data.struc, dtype=torch.float32)], dim=1)
        x1 = self.act(self.fc1(out))
        x1 = self.act(self.fc11(x1))
        x1 = self.act(self.fc12(x1))
        x1 = self.fc13(x1)

        return x1
