import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from sklearn.model_selection import train_test_split as tts

from src.Dataset import HiC_Dataset
from src.layers.WEGATConv import WEGATConv

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Sequential
import torch_geometric as tgm
from torch_geometric.data import DataLoader
from torch_geometric.nn import TopKPooling as TKP
from torch_geometric.nn import global_max_pool

from torch.optim.lr_scheduler import StepLR

OUTPATH = "/home/dh486/rds/hpc-work/GrapHiC-ML/Data/"
MODELOUTNAME = "edge_weighted_GAT_initial_train.pt"
TRAINACCOUTNAME = "train_accuracy"
TESTACCOUTNAME = "test_accuracy"
NUMEPOCHS = 2000
BATCHSIZE = 500
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 5e-4
RANDOM_STATE = 40
TEST_SIZE = 0.25

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

class WEGAT_Net(torch.nn.Module):
    def __init__(self, 
                 hidden_channels,
                 heads = 4,
                 num_fc = 5,
                 fc_channels = [15,15,10,5,2],
                 num_prom_fc = 5,
                 prom_fc_channels = [15,15,10,5,2]
                ):
        if isinstance(fc_channels,int):
            fc_channels = [fc_channels]*num_fc
        elif len(fc_channels) != num_fc:
            print("number of fully connected channels must match the number of fully connected layers")
            raise
            
        if isinstance(prom_fc_channels,int):
            prom_fc_channels = [prom_fc_channels]*num_prom_fc
        elif len(prom_fc_channels) != num_prom_fc:
            raise
        
        super(WEGAT_Net, self).__init__()
        torch.manual_seed(12345)

        self.loglikelihood_precision = Parameter(torch.tensor(0.))
        self.conv1 = WEGATConv(in_channels = NUMCHIP, 
                               node_out_channels = hidden_channels,
                               edge_channels = NUMEDGE,
                               edge_out_channels = NUMEDGE,
                               heads = heads,
                               concat = False
                              )
        self.pool1 = TKP(in_channels = hidden_channels)
        self.conv2 = WEGATConv(in_channels = hidden_channels,
                               node_out_channels = hidden_channels,
                               edge_channels = NUMEDGE,
                               edge_out_channels = NUMEDGE,
                               heads = heads,
                               concat = False
                              )
        self.pool2 = TKP(in_channels = hidden_channels)
        self.conv3 = WEGATConv(in_channels = hidden_channels,
                               node_out_channels = hidden_channels,
                               edge_channels = NUMEDGE,
                               edge_out_channels = NUMEDGE,
                               heads = heads,
                               concat = False
                              )
        self.pool3 = TKP(in_channels = hidden_channels)

        fc_channels = [hidden_channels]+fc_channels
        lin = []
        for idx in torch.arange(num_fc):
            lin.append(Linear(fc_channels[idx],fc_channels[idx+1]))
            lin.append(torch.nn.ReLU())

        self.lin = Sequential(*lin)
        self.num_fc = num_fc
        
        prom_fc_channels = [NUMCHIP]+prom_fc_channels
        linprom = []
        for idx in torch.arange(num_prom_fc):
            linprom.append(Linear(prom_fc_channels[idx],prom_fc_channels[idx+1]))
            linprom.append(torch.nn.ReLU())

        self.linprom = Sequential(*linprom)
        self.num_prom_fc = num_prom_fc
        
        
        self.readout = Linear(prom_fc_channels[-1]+fc_channels[-1], 1)

    def forward(self, 
                x,
                edge_index, 
                edge_attr,
                prom_x,
                batch):
        prom_x = prom_x.view(-1,NUMCHIP).float()
        
        edge_attr[torch.isnan(edge_attr)] = 0
        #superlayer 1
        x, edge_attr = self.conv1(x.float(), 
                                  edge_attr.float(),
                                  edge_index)
        #nonlinearity
        x = x.relu()
        edge_attr.relu()
        #pooling
        x, edge_index, edge_attr, batch,perm,score = self.pool1(x,
                                                                edge_index,
                                                                edge_attr = edge_attr,
                                                                batch = batch)
                                                                 
        #superlayer 2
        x, edge_attr = self.conv2(x,
                                  edge_attr,
                                  edge_index)
        #nonlinearity
        x = x.relu()
        edge_attr.relu()
        #pooling
        x, edge_index, edge_attr,batch,perm,score = self.pool2(x,
                                                               edge_index,
                                                               edge_attr = edge_attr,
                                                               batch = batch)
        
        #superlayer 3
        x,_ = self.conv3(x,
                        edge_attr,
                        edge_index)
        #nonlinearity
        x = x.relu()
        edge_attr = edge_attr.relu()
        #pooling
        x, edge_index, edge_attr, batch, perm,score = self.pool3(x,
                                                                 edge_index,
                                                                 edge_attr = edge_attr,
                                                                 batch = batch)

        #global pooling
        x = global_max_pool(x,batch=batch)

        # 3. Apply fully connected linear layers to graph
        x = self.lin(x)
        
        # 3. Apply fully connected linear layers to promoter
        prom_x = self.linprom(prom_x)
        
        # 4. Apply readout layers 
        x = self.readout(torch.cat([x,prom_x],
                                   dim = 1)
                        )
        
        return x



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

#Hack for now to just load everything into memory since the graphs aren't massive
print("Loading datasets into working memory")
#dset1 = [torch.load(f).to(device) for f in glob.glob("Data/processed/train/*")]
#dset2 = [torch.load(f).to(device) for f in glob.glob("Data/processed/test/*")]
#dset = dset1+dset2

dset = torch.load("Data/test_dset_18features_custom_norm.pt")

train_dset, test_dset,_,_ = tts(dset,
                                  torch.ones(len(dset)),
                                  test_size=TEST_SIZE,
                                  random_state=RANDOM_STATE)


train_dset = [item.to(device) for item in train_dset]
test_dset = [item.to(device) for item in test_dset]

print("Loaded in memory datasets")
train_loader = DataLoader(train_dset, 
                          batch_size=BATCHSIZE)
test_loader = DataLoader(test_dset, 
                         batch_size=BATCHSIZE)

model = WEGAT_Net(hidden_channels = 18).to(device)

optimizer = torch.optim.Adam(model.parameters(),
                             lr = LEARNING_RATE,
                             weight_decay = WEIGHT_DECAY)

criterion = torch.nn.L1Loss()


def train(loader, 
          model, 
          optimizer, 
          criterion):
    model.train()
    accs = []
    precision = torch.exp(model.loglikelihood_precision)
    for data in loader:
        out = model(data.x, 
                data.edge_index, 
                data.edge_attr,
                data.prom_x,
                data.batch)  # Perform a single forward pass.
        loss = precision*criterion(out[:,0],data.y.float()) - BATCHSIZE*model.loglikelihood_precision  # Compute the loss.
        loss.backward(retain_graph=True)  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad() # Clear gradients.
        accs.append(loss.item())
    
    return accs

def test(loader, 
         model, 
         criterion):
    model.eval()
    
    accs = []
    precision = torch.exp(model.loglikelihood_precision)
    for data in loader:
        pred = model(data.x, 
                data.edge_index, 
                data.edge_attr,
                data.prom_x,
                data.batch)
        acc = precision*criterion(pred[:,0],data.y.float()) - BATCHSIZE*model.loglikelihood_precision
        accs.append(acc.item())
        
    return accs


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
                     np.mean(trainacc),
                     np.mean(testacc)
                   ))

    
    #save the updated model and running accuracies
    torch.save(model.state_dict(), os.path.join(OUTPATH,MODELOUTNAME))
    np.save(os.path.join(OUTPATH,TRAINACCOUTNAME), train_accs)
    np.save(os.path.join(OUTPATH,TESTACCOUTNAME), test_accs)
