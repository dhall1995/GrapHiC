import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
import networkx as nx

from src.Dataset import HiC_Dataset
from src.layers.WEGATConv import WEGATConv

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
from torch_geometric.nn import global_mean_pool
import torch_geometric as tgm


bigwigs = os.listdir("Data/raw/bigwigs")
contacts = os.listdir("Data/raw/contacts")
target = "target.tsv"

train_dset = HiC_Dataset("Data",
                   contacts=contacts,
                   bigwigs=bigwigs,
                   target=target)

NUMCHIP = train_dset.num_node_features
NUMEDGE = train_dset.num_edge_features

class WEGAT_Net(torch.nn.Module):
    def __init__(self, hidden_channels):
        super(WEGAT_Net, self).__init__()
        torch.manual_seed(12345)
        self.conv1 = WEGATConv(in_channels = NUMCHIP, 
                            node_out_channels = hidden_channels, 
                            edge_channels = NUMEDGE,
                            edge_out_channels = NUMEDGE
                           )
        self.conv2 = WEGATConv(in_channels = hidden_channels, 
                            node_out_channels = hidden_channels, 
                            edge_channels = NUMEDGE,
                            edge_out_channels = NUMEDGE
                           )
        self.conv3 = WEGATConv(in_channels = hidden_channels, 
                            node_out_channels = hidden_channels, 
                            edge_channels = NUMEDGE,
                            edge_out_channels = NUMEDGE
                           )
        self.lin = Linear(hidden_channels, 1)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings 
        x, edge_attr = self.conv1(x.float(), edge_attr.float(), edge_index)
        x = x.relu()
        edge_attr.relu()
        x, edge_attr = self.conv2(x, edge_attr,edge_index)
        x = x.relu()
        edge_attr.relu()
        x,_ = self.conv3(x, edge_attr,edge_index)

        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]

        # 3. Apply a final classifier
        x = self.lin(x)
        
        return x



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = WEGAT_Net().to(device)
train_dset = train_dset.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)


def train():
    model.train()
    optimizer.zero_grad()
    criterion = torch.nn.MSELoss()
    criterion(model(), y).backward()
    optimizer.step()



def test():
    model.eval()
    logits, accs = model(), []
    for _, mask in data('train_mask', 'val_mask', 'test_mask'):
        pred = logits[mask]
        criterion = torch.nn.MSELoss()
        acc = criterion(pred, y[mask,None])
        accs.append(acc)
    return accs


all_accs = []
for epoch in range(1, 3):
    train()
    log = 'Epoch: {:03d}, Train: {:.4f}, Val: {:.4f}, Test: {:.4f}'
    all_accs.append('{:.4f} {:.4f} {:.4f}'.format(*test()))
    if epoch%500 == 0 or epoch == 1:
        print(log.format(epoch, *test()))
        
torch.save(model.state_dict(), "/home/dh486/rds/hpc-work/GNN_Work/test_model")
