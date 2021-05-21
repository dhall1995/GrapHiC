import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from math import pi as PI
from collections import OrderedDict

from src.Dataset import HiC_Dataset
from src.layers.WEGATConv import WEGAT_TOPK_Conv
from src.layers.utils import PositionalEncoding

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Sequential, Dropout
from torch.utils.data import random_split

import torch_geometric as tgm
from torch_geometric.data import DataLoader
from torch_geometric.nn import global_max_pool

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

DATASET = "Data/test_dset_18features_custom_norm.pt"
NUMEPOCHS = 10
BATCHSIZE = 200
LEARNING_RATE = 0.00005
DROPOUT = 0.1
MANUAL_SEED = 40

'''
UTILITY FUNCTIONS
'''
def get_middle_features(x, 
                        numnodes = 51
                       ):
    mid = int((numnodes-1)/2)
    idxs = torch.arange(mid, x.shape[0], numnodes)
    return x[idxs,:]


'''
WEIGHTED EDGE GRAPH ATTENTION MODULE
'''
class WEGATModule(torch.nn.Module):
    def __init__(self, 
                 hidden_channels=25,
                 numchip = 18,
                 numedge = 3,
                 heads = 4,
                 num_graph_convs = 4,
                 num_fc = 5,
                 fc_channels = [15,15,10,5,2],
                 num_prom_fc = 8,
                 prom_fc_channels = [15,15,10,10,10,10,5,2],
                 positional_encoding = True,
                 dropout = 0.1
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
        
        #dropout layer
        self.dropout = Dropout(p=dropout)
        
        #number of input chip features
        self.numchip = numchip
        
        #Whether to apply positional encoding to nodes
        self.positional_encoding = positional_encoding
        if positional_encoding:
            self.posencoder = PositionalEncoding(hidden_channels, 
                                                 dropout=dropout)
        
        #initial embeddding layer
        self.embedding = Sequential(OrderedDict([
            ('lin_embedding', Linear(numchip, hidden_channels)),
            ('lin_relu', torch.nn.ReLU())
        ]))
        
        #graph convolution layers
        gconv = []
        for idx in np.arange(num_graph_convs):
            gconv.append(WEGAT_TOPK_Conv(node_inchannels = hidden_channels,
                                     node_outchannels = hidden_channels,
                                     edge_inchannels = numedge,
                                     edge_outchannels = numedge,
                                     heads = heads
                                    )
                        )

        self.gconv = Sequential(*gconv)
        
        #fully connected channels
        fc_channels = [hidden_channels]+fc_channels
        lin = []
        for idx in torch.arange(num_fc):
            lin.append(Linear(fc_channels[idx],fc_channels[idx+1]))
            lin.append(torch.nn.ReLU())
        self.lin = Sequential(*lin)
        self.num_fc = num_fc
        
        #fully connected promoter channels 
        prom_fc_channels = [numchip]+prom_fc_channels
        linprom = []
        for idx in torch.arange(num_prom_fc):
            linprom.append(Linear(prom_fc_channels[idx],prom_fc_channels[idx+1]))
            linprom.append(torch.nn.ReLU())
        self.linprom = Sequential(*linprom)
        self.num_prom_fc = num_prom_fc
        
        #final readout function
        self.readout = Linear(prom_fc_channels[-1]+fc_channels[-1], 1)
        
    def forward(self, 
                batch):
        batch.prom_x = batch.prom_x.view(-1,self.numchip).float()
        batch.edge_attr[torch.isnan(batch.edge_attr)] = 0
        batch.x[torch.isnan(batch.x)] = 0
        batch.prom_x[torch.isnan(batch.prom_x)] = 0
        
        #initial dropout and embedding
        batch.x = self.dropout(batch.x)
        batch.x = self.embedding(batch.x.float())
        
        #positional encoding
        if self.positional_encoding:
            batch.x = self.posencoder(batch.x, 
                                      batch.batch)
        #graph convolutions
        batch = self.gconv(batch)

        #global pooling
        #x = global_max_pool(batch.x,
        #                    batch=batch.batch)
        x = self.dropout(batch.x)
        x = get_middle_features(x)

        # 3. Apply fully connected linear layers to graph
        x = self.lin(x)
        
        # 3. Apply fully connected linear layers to promoter
        prom_x = self.dropout(batch.prom_x)
        prom_x = self.linprom(prom_x)
        
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
                 train_loader,
                 val_loader,
                 learning_rate
                ):
        super().__init__()
        self.WEGATModule = module
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.val_loader = val_loader
    
    def train_dataloader(self):
        return self.train_loader
    def validation_dataloader(self):
        return self.test_loader


    def shared_step(self, batch):
        pred = self.WEGATModule(batch)
        
        loss = F.l1_loss(pred[:,0],
                         batch.y.float())
        return loss, pred[:,0]
    
    def customlog(self, name, loss, pred):
        self.log(f'{name}_loss', loss)
        self.log(f'{name}_maxabs_prediction', 
                 torch.max(abs(pred)).item())
        self.log(f'{name}_mean_prediction', 
                 torch.mean(pred).item())
        self.log(f'{name}_std_prediction', 
                 torch.std(pred).item())
        
    def training_step(self, batch, batch_idx):
        loss, pred = self.shared_step(batch)
        self.customlog('train',loss, pred)
        return loss
    
    def validation_step(self, batch, batch_idx):
        loss, pred = self.shared_step(batch)
        self.customlog('val',loss, pred)
        return loss
    
    def test_step(self, batch, batch_idx):
        loss, pred = self.shared_step(batch)
        self.customlog('test',loss, pred)
        return loss     
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), 
                                     lr=self.learning_rate)
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
    NUMNODESPERGRAPH = dset[0].x.shape[0]
    
    module = WEGATModule(hidden_channels = hparams.hiddenlayers,
                         numchip = NUMCHIP,
                         numedge = NUMEDGE,
                         positional_encoding = hparams.positional_encoding,
                         dropout = hparams.dropout
                        )
    Net = LitWEGATNet(module,
                      train_loader,
                      val_loader,
                      hparams.learning_rate)

    tb_logger = pl_loggers.TensorBoardLogger(hparams.logdir)
    trainer = pl.Trainer(gpus=hparams.gpus, 
                         max_epochs=hparams.epochs, 
                         progress_bar_refresh_rate=1,
                         logger=tb_logger,
                         auto_lr_find=hparams.auto_lr_find,
                         resume_from_checkpoint=hparams.checkpoint
                         )
    if hparams.auto_lr_find:
        # Run learning rate finder
        lr_finder = trainer.tuner.lr_find(Net)

        # Results can be found in
        lr_finder.results

        # Plot with
        if hparams.plot_lr:
            fig = lr_finder.plot(suggest=True)
            fig.savefig("learning_rate_suggestion.png", format = 'png')

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()

        # update hparams of the model
        Net.hparams.lr = new_lr

    trainer.fit(Net, train_loader, val_loader)
    

if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-g',
                        '--gpus',
                        type = int,
                        default=1)
    parser.add_argument('-hidden',
                        '--hiddenlayers',
                        type = int,
                        default=15)
    parser.add_argument('-e',
                        '--epochs',
                        type = int,
                        default=NUMEPOCHS)
    parser.add_argument('-l',
                        '--logdir',
                        type = str,
                        default='runs/')
    parser.add_argument('-b',
                        '--batchsize',
                        type = int,
                        default=BATCHSIZE)
    parser.add_argument('-d',
                        '--dataset',
                        type = str,
                        default=DATASET)
    parser.add_argument('-t',
                        '--trainfraction',
                        type = float,
                        default=0.7)
    parser.add_argument('--learning_rate',
                        type = float,
                        default=LEARNING_RATE)
    parser.add_argument('--dropout',
                        type = float,
                        default=DROPOUT)
    parser.add_argument('-c',
                        '--checkpoint',
                        default=None)
    parser.add_argument('-alr',
                        '--auto_lr_find',
                        type = bool,
                        default=False)
    parser.add_argument('-plr',
                        '--plot_lr',
                        type = bool,
                        default=False)
    parser.add_argument('-p',
                        '--positional_encoding',
                        type = bool,
                        default=True)
    args = parser.parse_args()

    main(args)            
            
            
