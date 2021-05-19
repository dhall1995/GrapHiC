import numpy as np
import matplotlib.pyplot as plt
import glob
import os

from src.Dataset import HiC_Dataset
from src.layers.WEGATConv import WEGATConv

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Sequential
from torch.utils.data import random_split

import torch_geometric as tgm
from torch_geometric.data import DataLoader
from torch_geometric.nn import TopKPooling as TKP
from torch_geometric.nn import global_max_pool

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

DATASET = "Data/test_dset_18features_custom_norm.pt"
NUMEPOCHS = 10
BATCHSIZE = 500
LEARNING_RATE = 0.00005
WEIGHT_DECAY = 5e-4
MANUAL_SEED = 40

'''
COMBINED 'WEIGHTED EDGE GRAPH ATTENTION' + 'TOP K POOLING LAYERS' 
'''
class WEGAT_TOPK_Conv(torch.nn.Module):
    def __init__(self,
                 node_inchannels,
                 node_outchannels,
                 edge_inchannels,
                 edge_outchannels,
                 heads = 4):
        super().__init__()
        self.conv = WEGATConv(in_channels = node_inchannels, 
                               node_out_channels = node_outchannels,
                               edge_channels = edge_inchannels,
                               edge_out_channels = edge_outchannels,
                               heads = heads,
                               concat = False
                              )
        self.pool = TKP(in_channels = node_outchannels)
        
    def forward(self, 
                batch):
        batch.x, batch.edge_attr = self.conv(batch.x.float(),
                                             batch.edge_attr.float(),
                                             batch.edge_index)
        batch.x = batch.x.relu()
        batch.edge_attr = batch.edge_attr.relu()
        batch.x, batch.edge_index, batch.edge_attr, batch.batch, perm,score = self.pool(batch.x,
                                                                                        batch.edge_index,
                                                                                        edge_attr = batch.edge_attr,
                                                                                        batch = batch.batch)
        
        return batch
        


'''
WEIGHTED EDGE GRAPH ATTENTION MODULE
'''
class WEGATModule(torch.nn.Module):
    def __init__(self, 
                 hidden_channels,
                 numchip = 18,
                 numedge = 3,
                 heads = 4,
                 num_graph_convs = 3,
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
            
        if num_graph_convs < 1:
            print("need at least one graph convolution")
            raise
        num_graph_convs = int(num_graph_convs)
            
        if isinstance(prom_fc_channels,int):
            prom_fc_channels = [prom_fc_channels]*num_prom_fc
        elif len(prom_fc_channels) != num_prom_fc:
            raise
        
        super().__init__()
        torch.manual_seed(12345)

        self.loglikelihood_precision = Parameter(torch.tensor(0.))
        gconv = [WEGAT_TOPK_Conv(node_inchannels = numchip, 
                             node_outchannels = hidden_channels,
                             edge_inchannels = numedge,
                             edge_outchannels = numedge,
                             heads = heads
                            )
                ]
        for idx in np.arange(num_graph_convs-1):
            gconv.append(WEGAT_TOPK_Conv(node_inchannels = hidden_channels,
                                     node_outchannels = hidden_channels,
                                     edge_inchannels = numedge,
                                     edge_outchannels = numedge,
                                     heads = heads
                                    )
                        )

        self.gconv = Sequential(*gconv)

        fc_channels = [hidden_channels]+fc_channels
        lin = []
        for idx in torch.arange(num_fc):
            lin.append(Linear(fc_channels[idx],fc_channels[idx+1]))
            lin.append(torch.nn.ReLU())

        self.lin = Sequential(*lin)
        self.num_fc = num_fc
        self.numchip = numchip
        
        prom_fc_channels = [numchip]+prom_fc_channels
        linprom = []
        for idx in torch.arange(num_prom_fc):
            linprom.append(Linear(prom_fc_channels[idx],prom_fc_channels[idx+1]))
            linprom.append(torch.nn.ReLU())

        self.linprom = Sequential(*linprom)
        self.num_prom_fc = num_prom_fc
        
        
        self.readout = Linear(prom_fc_channels[-1]+fc_channels[-1], 1)
        
    def forward(self, 
                batch):
        batch.prom_x = batch.prom_x.view(-1,self.numchip).float()
        batch.edge_attr[torch.isnan(batch.edge_attr)] = 0
        batch.x[torch.isnan(batch.x)] = 0
        batch.prom_x[torch.isnan(batch.prom_x)] = 0
        
        batch = self.gconv(batch)

        #global pooling
        x = global_max_pool(batch.x,
                            batch=batch.batch)

        # 3. Apply fully connected linear layers to graph
        x = self.lin(x)
        
        # 3. Apply fully connected linear layers to promoter
        prom_x = self.linprom(batch.prom_x)
        
        # 4. Apply readout layers 
        x = self.readout(torch.cat([x,prom_x],
                                   dim = 1)
                        )
        
        return x

'''
LIGHTNING NET
'''
class LitWEGATNet(pl.LightningModule):
    def __init__(self,
                 module,
                 learning_rate,
                 weight_decay
                ):
        super().__init__()
        self.WEGATModule = module
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
    
    def shared_step(self, batch):
        pred = self.WEGATModule(batch)
        
        loss = F.l1_loss(pred[:,0],
                         batch.y.float())
        return loss
    
    def training_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss = self.shared_step(batch)
        self.log('test_loss', loss)
        return loss     
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.learning_rate,
                                     weight_decay = self.weight_decay
                                    )
        return optimizer
    
    
'''
MAIN FUNCTION
'''    
def main(hparams):
    '''
    CONSTRUCTING THE DATALOADERS
    '''
    print("Loading in memory datasets")
    dset = torch.load(hparams.dataset)

    numdatapoints = len(dset)
    trainsize = int(numdatapoints*hparams.trainfraction)
    train_dset, val_dset = random_split(dset,
                                        [trainsize, numdatapoints-trainsize],
                                        generator=torch.Generator().manual_seed(MANUAL_SEED)
                                       )

    print("Loaded in memory datasets")
    train_loader = DataLoader(train_dset, 
                              batch_size=hparams.batchsize
                             )
    val_loader = DataLoader(val_dset, 
                             batch_size=hparams.batchsize
                           )


    '''
    INITIALISING/TRAINING THE MODEL
    '''
    NUMCHIP = dset[0].x.shape[1]
    NUMEDGE = dset[0].edge_attr.shape[1]
    
    module = WEGATModule(hidden_channels = hparams.hiddenlayers,
                      numchip = NUMCHIP,
                      numedge = NUMEDGE
                     )
    Net = LitWEGATNet(module, 
                      hparams.learning_rate, 
                      hparams.weight_decay)

    tb_logger = pl_loggers.TensorBoardLogger(hparams.logdir)
    trainer = pl.Trainer(gpus=hparams.gpus, 
                         max_epochs=hparams.epochs, 
                         progress_bar_refresh_rate=20,
                         logger=tb_logger)
    trainer.fit(Net, train_loader, val_loader)
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-g',
                        '--gpus', 
                        default=1)
    parser.add_argument('-hidden',
                        '--hiddenlayers',
                        default=15)
    parser.add_argument('-e',
                        '--epochs',
                        default=NUMEPOCHS)
    parser.add_argument('-l',
                        '--logdir',
                        default='runs/')
    parser.add_argument('-b',
                        '--batchsize',
                        default=BATCHSIZE)
    parser.add_argument('-d',
                        '--dataset',
                        default=DATASET)
    parser.add_argument('-t',
                        '--trainfraction',
                        default=0.7)
    parser.add_argument('--learning_rate',
                        default=LEARNING_RATE)
    parser.add_argument('--weight_decay',
                        default=WEIGHT_DECAY)
    args = parser.parse_args()

    main(args)            
            
            
