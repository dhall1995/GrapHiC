import numpy as np
from ..layers.GATEConv import Deep_GATEv2_Conv
from ..layers.utils import PositionalEncoding
from ..utils.misc import get_middle_features

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Sequential, Dropout,BatchNorm1d,ModuleList, ReLU

NUMCLASSES = 3
    
'''
WEIGHTED EDGE GRAPH ATTENTION MODULE
'''
class GrapHiC_Encoder(torch.nn.Module):
    def __init__(self,
                 hidden_channels=20,
                 numchip = 18,
                 numedge = 2,
                 heads = 4,
                 num_graph_convs = 5,
                 embedding_layers = 2,
                 num_fc = 5,
                 fc_channels = [15,15,10,10,5],
                 positional_encoding = True,
                 pos_embedding_dropout = 0.1,
                 fc_dropout = 0.5,
                 conv_dropout = 0.1,
                 numclasses = NUMCLASSES
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

        super().__init__()
        torch.manual_seed(12345)
        #dropout layer
        self.dropout = Dropout(p=fc_dropout)

        #number of input chip features
        self.numchip = numchip

        #Whether to apply positional encoding to nodes
        self.positional_encoding = positional_encoding
        if positional_encoding:
            self.posencoder = PositionalEncoding(hidden_channels,
                                                 dropout=pos_embedding_dropout,
                                                 identical_sizes = True
                                                )

        #initial embeddding layer
        embedding = []
        embedding.append(Sequential(Linear(numchip,hidden_channels),
                                    Dropout(p=fc_dropout),
                                    BatchNorm1d(hidden_channels),
                                    ReLU())
                        )
        for idx in torch.arange(embedding_layers - 1):
            embedding.append(Sequential(Linear(hidden_channels,hidden_channels),
                                         Dropout(p=fc_dropout),
                                         BatchNorm1d(hidden_channels),
                                         ReLU())
                            )
        self.embedding = ModuleList(embedding)

        #graph convolution layers
        #Encoding layer
        enc = Deep_GATEv2_Conv(node_in_channels = hidden_channels,
                                node_out_channels = hidden_channels,
                                edge_in_channels = numedge,
                                edge_out_channels = numedge,
                                heads = heads,
                                node_dropout = conv_dropout,
                                edge_dropout = conv_dropout
                               )
        gconv = [enc]
        #encoder/decoder layer
        encdec = Deep_GATEv2_Conv(node_in_channels = hidden_channels,
                                   node_out_channels = hidden_channels,
                                   edge_in_channels = numedge,
                                   edge_out_channels = numedge,
                                   heads = heads,
                                   node_dropout = conv_dropout,
                                   edge_dropout = conv_dropout
                                    )
        for idx in np.arange(num_graph_convs-1):
            gconv.append(encdec)

        self.gconv = Sequential(*gconv)

        #fully connected channels
        fc_channels = [hidden_channels]+fc_channels
        lin = []
        for idx in torch.arange(num_fc):
            lin.append(Linear(fc_channels[idx],fc_channels[idx+1]))
            lin.append(torch.nn.Dropout(p=fc_dropout))
            lin.append(BatchNorm1d(fc_channels[idx+1]))
            lin.append(torch.nn.ReLU())
        self.lin = Sequential(*lin)
        
        #final readout function
        self.readout = Linear(fc_channels[-1], numclasses)

    def forward(self,
                batch):
        batch.prom_x = batch.prom_x.view(-1,self.numchip).float()
        batch.edge_attr[torch.isnan(batch.edge_attr)] = 0
        batch.x[torch.isnan(batch.x)] = 0
        batch.prom_x[torch.isnan(batch.prom_x)] = 0
        
        #initial dropout and embedding
        batch.x = self.dropout(batch.x)
        batch.prom_x = self.dropout(batch.prom_x)
        batch.x = self.embedding[0](batch.x.float())
        batch.prom_x = self.embedding[0](batch.prom_x.float())
        for mod in self.embedding[1:]:               
            batch.x = batch.x + mod(batch.x.float())
            batch.prom_x = batch.prom_x + mod(batch.prom_x.float())

        #positional encoding
        if self.positional_encoding:
            batch.x = self.posencoder(batch.x,
                                      batch.batch)

        #graph convolutions
        batch = self.gconv(batch)

        #extracting node of interest from graph
        batch.x = get_middle_features(batch.x)
        
        #combining node-level and promoter level representations
        batch.x += batch.prom_x

        # 3. Apply fully connected linear layers
        batch.x = self.lin(batch.x)

        # 4. Apply readout layers
        x = self.readout(batch.x)

        return x
