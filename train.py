import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

from src.Dataset import HiC_Dataset
from src.layers.WEGATConv import WEGATConv

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear
import torch_geometric as tgm
from torch_geometric.data import DataLoader

OUTPATH = "/home/dh486/rds/hpc-work/GrapHiC-ML/Data/"
MODELOUTNAME = "edge_weighted_GAT_initial_train.pt"
TRAINACCOUTNAME = "train_accuracy"
TESTACCOUTNAME = "test_accuracy"
NUMEPOCHS = 500
BATCHSIZE = 500
LEARNING_RATE = 0.005
WEIGHT_DECAY = 5e-4


bigwigs = os.listdir("Data/raw/bigwigs")
contacts = os.listdir("Data/raw/contacts")
target = "target.tsv"

print("Making datasets")
train_dset = HiC_Dataset("Data",
                         contacts=contacts,
                         bigwigs=bigwigs,
                         target=target
                        )

test_dset = HiC_Dataset("Data",
                         contacts=contacts,
                         bigwigs=bigwigs,
                         target=target,
                         train=False
                        )

print("Made out-of-working-memory datasets")

NUMCHIP = train_dset.num_node_features
NUMEDGE = train_dset.num_edge_features
NUMNODESPERGRAPH = train_dset.numnodespergraph

def get_middle_features(x,
                        numnodes_per_graph):
    mid = int(0.5*(numnodes_per_graph-1))
    idxs = np.arange(mid,
                     x.shape[0],
                     numnodes_per_graph)
    return x[idxs,:]

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

    def forward(self, 
                x, 
                edge_index, 
                edge_attr, 
                batch):
        edge_attr[torch.isnan(edge_attr)] = 0
        # 1. Obtain node embeddings 
        x, edge_attr = self.conv1(x.float(), 
                                  edge_attr.float(),
                                  edge_index)
        x = x.relu()
        edge_attr.relu()
        x, edge_attr = self.conv2(x, 
                                  edge_attr,
                                  edge_index)
        x = x.relu()
        edge_attr.relu()
        x,_ = self.conv3(x, 
                         edge_attr,
                         edge_index)

        # 2. Readout layer
        x = get_middle_features(x, 
                                NUMNODESPERGRAPH)
        
        # 3. Apply a final classifier
        x = self.lin(x)
        
        return x



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device:{device}")
model = WEGAT_Net(hidden_channels = 30).to(device)
optimizer = torch.optim.Adam(model.parameters(), 
                             lr=LEARNING_RATE, 
                             weight_decay=WEIGHT_DECAY)
criterion = torch.nn.MSELoss()

#Hack for now to just load everything into memory since the graphs aren't massive
print("Loading datasets into working memory")
train_dset = [torch.load(f).to(device) for f in glob.glob("Data/processed/train/*")]
test_dset = [torch.load(f).to(device) for f in glob.glob("Data/processed/test/*")]

print("Loaded in memory datasets")
train_loader = DataLoader(train_dset, 
                          batch_size=BATCHSIZE)
test_loader = DataLoader(test_dset, 
                         batch_size=BATCHSIZE)
def train(loader, 
          model, 
          optimizer, 
          criterion):
    model.train()
    accs = []
    for data in loader:
        out = model(data.x, 
                data.edge_index, 
                data.edge_attr,
                data.batch)  # Perform a single forward pass.
        loss = criterion(10*out[:,0],
                         10*data.y.float())  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad() # Clear gradients.
        accs.append(loss.item())
    
    return np.mean(accs)

def test(loader, 
         model, 
         criterion):
    model.eval()
    
    accs = []
    for data in loader:
        pred = model(data.x, 
                data.edge_index, 
                data.edge_attr,
                data.batch) 
        acc = criterion(10*pred[:,0], 
                        10*data.y.float())
        accs.append(acc.item())
        
    return np.mean(accs)


train_accs = []
test_accs = []
print("Running training...:")
for epoch in range(1, NUMEPOCHS+1):
    log = 'Epoch: {:03d}, Train: {:.4f}, Test: {:.4f}'
    trainacc = train(train_loader, 
                     model, 
                     optimizer, 
                     criterion)
    testacc = test(test_loader,
                    model,
                    criterion
                   )
    train_accs.append(trainacc)
    test_accs.append(testacc)
    print(log.format(epoch, 
                     trainacc,
                     testacc
                    ))
    
    #save the updated model and running accuracies
    torch.save(model.state_dict(), os.path.join(OUTPATH,MODELOUTNAME))
    np.save(os.path.join(OUTPATH,TRAINACCOUTNAME), train_accs)
    np.save(os.path.join(OUTPATH,TESTACCOUTNAME), test_accs)
