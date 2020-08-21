import os.path as osp
import argparse

import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from layers.GATconv import GATconv_Edge_Weighted as GATCEW

from utils.datasets import pop_HiC_Dataset
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--use_gdc', action='store_true',
                    help='Use Graph Diffusion Convolution preprocessing.')
args = parser.parse_args()

path = "/home/dh486/rds/hpc-work/GNN_Work/data/"

popfeatures = ['2i_DNA_meth.npz',
 'GSE22562_Med12_ESC_peaks.npz',
 'naive_mESC_poised_enhancers.npz',
 'H3K4me3_hapmESC_highconfpeaks.npz',
 'H3K4me1_hapmESC_highconfpeaks.npz',
 'naive_mESC_active_promoters.npz',
 'Cbx7_ES.npz',
 'GSE56098_Oct4_ESC_peaks.npz',
 'TSS_sites.npz',
 'naive_mESC_inactive_promoters.npz',
 'Cbx3_ES.npz',
 'Klf4_hapmESC_highconfpeaks.npz',
 'CTCF_highconfpeaks.npz',
 'mESC_std_nuclear_depth.npz',
 'mESC_mean_nuclear_depth.npz',
 'GSE22562_Med1_ESC_peaks.npz',
 'H3K27ac_hapmESC_highconfpeaks.npz',
 'naive_mESC_bivalent_promoters.npz',
 'CGI_mm10.npz',
 'GSE56098_Otx2_ESC_peaks.npz',
 'GSEXXXXXHendrich20150626_Chd4_ESC_peaks.npz',
 'H3K9me3_hapmESC_highconfpeaks.npz',
 'mESC_mean_local_RoG2Mb.npz',
 'naive_mESC_active_enhancers.npz',
 'GSEXXXXXHendrich20150626_Mbd3_ESC_peaks.npz',
 'Nanog_hapmESC_highconfpeaks.npz',
 'mESC_mean_chromosomal_depth.npz',
 'mESC_std_local_RoG2Mb.npz',
 'H3K27me3_hapmESC_highconfpeaks.npz',
 'mESC_std_chromosomal_depth.npz',
 'Ring1b_ES.npz',
 'H3K36me3_hapmESC_highconfpeaks.npz',
 'GSE56098_p300_ESC_peaks.npz',
 'Smc3_hapmESC_highconfpeaks.npz',
 'naive_mESC_highly_active_promoters.npz']

dset = pop_HiC_Dataset(path,
                       condition = 'SLX-7671_haploid',
                       features = {item:None for item in popfeatures},
                       target = {'totalRNA.npz':None},
                       targetIDs = False,
                       target_mask = True,
                       binSize = 50
                      )
data = dataset[0]

if args.use_gdc:
    attr = data.edgeattr
    data.edgeattr = data.edgeattr[:,0]
    gdc = T.GDC(self_loop_weight=1, normalization_in='sym',
                normalization_out='col',
                diffusion_kwargs=dict(method='ppr', alpha=0.05),
                sparsification_kwargs=dict(method='topk', k=128,
                                           dim=0), exact=True)
    data = gdc(data)
    data.edgeattr = attr
    
'''
class MyNet(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = GATCEW(dataset.num_features, 6, heads=8, dropout=0.6)
        self.conv2 = GATCEW(6 * 8, 4, heads=8, concat=True, dropout=0.6)
        self.fc1 = torch.nn.Linear(32, 40)
        self.fc2 = torch.nn.Linear(40, 10)
        self.linear = torch.nn.Linear(10, 1)

    def forward(self):
        #Graph Attention Layer 1
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, data.edge_index, edge_attr = data.edge_attr))
        
        #Graph Attention Layer 2
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.relu(self.conv2(x, data.edge_index, edge_attr = data.edge_attr))
        
        #Fully Connected Layes
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.linear(x)
        
        return x
'''



class MyNet(torch.nn.Module):
    def __init__(self):
        super(MyNet, self).__init__()
        self.conv1 = GATCEW(data.num_features, 6, heads=8,concat = True, dropout=0.6)
        self.fc1 = torch.nn.Linear(48, 10)
        self.linear = torch.nn.Linear(10, 1)

    def forward(self):
        #Graph Attention Layer 1
        x = F.dropout(data.x, p=0.6, training=self.training)
        x = F.relu(self.conv1(x, data.edge_index, edge_attr = data.edge_attr[:,0]))
        
        #Fully Connected Layes
        x = F.relu(self.fc1(x))
        x = self.linear(x)
        
        return x


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model, data = MyNet().to(device), data.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005, weight_decay=5e-4)

def scale(y):
    
    x = torch.zeros(y.shape)
    x[y>1] = 1 + np.log(y[y>1])
    x[y<1] = y[y<1]
    
    return x

y = scale(data.y)


def train():
    model.train()
    optimizer.zero_grad()
    criterion = torch.nn.MSELoss()
    criterion(model()[data.train_mask], y[data.train_mask,None]).backward()
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
