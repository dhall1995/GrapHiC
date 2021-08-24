import numpy as np
from ..layers.GATEConv import Deep_GATE_Conv
from ..layers.utils import PositionalEncoding
from ..utils.misc import get_middle_features

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Sequential, Dropout,BatchNorm1d,ModuleList, ReLU
from torch import sigmoid

from torch_geometric.utils import dropout_adj

NUMCLASSES = 3
NUMNODESPERGRAPH = 201
    
'''
WEIGHTED EDGE GRAPH ATTENTION MODULE
'''
class GATE_module(torch.nn.Module):
    def __init__(self,
                 hidden_channels=20,
                 inchannels = 15,
                 edgechannels = 2,
                 heads = 5,
                 num_graph_convs = 5,
                 embedding_layers = 2,
                 num_fc = 5,
                 fc_channels = 15,
                 positional_encoding = True,
                 pos_embedding_dropout = 0.1,
                 fc_dropout = 0.5,
                 edge_dropout = 0.1,
                 recurrent = True,
                 parameter_efficient = True,
                 attention_channels = None,
                 principal_neighbourhood_aggregation = False,
                 deg = None,
                 aggr = 'add'
                ):
        if num_graph_convs < 1:
            print("need at least one graph convolution")
            raise
        num_graph_convs = int(num_graph_convs)

        super().__init__()
        torch.manual_seed(12345)
        self.edge_dropout = edge_dropout

        #number of input chip features
        self.inchannels = inchannels

        #Whether to apply positional encoding to nodes
        self.positional_encoding = positional_encoding
        if positional_encoding:
            self.posencoder = PositionalEncoding(hidden_channels,
                                                 dropout=pos_embedding_dropout,
                                                 identical_sizes = True
                                                )

        #initial embeddding layer
        embedding = []
        embedding.append(Sequential(BatchNorm1d(inchannels),
                                    ReLU(),
                                    Dropout(p=fc_dropout),
                                    Linear(inchannels,
                                           hidden_channels))
                        )
        for idx in torch.arange(embedding_layers - 1):
            embedding.append(Sequential(BatchNorm1d(hidden_channels),
                                        ReLU(),
                                        Dropout(p=fc_dropout),
                                        Linear(hidden_channels,
                                               hidden_channels))
                            )
        self.embedding = ModuleList(embedding)

        #graph convolution layers
        #Encoding layer
        enc = Deep_GATE_Conv(node_in_channels = hidden_channels,
                             node_out_channels = hidden_channels,
                             edge_in_channels = edgechannels,
                             edge_out_channels = edgechannels,
                             heads = heads,
                             dropout = fc_dropout,
                             attention_channels = attention_channels,
                             parameter_efficient = parameter_efficient,
                             principal_neighbourhood_aggregation = principal_neighbourhood_aggregation,
                             deg = deg,
                             aggr=aggr
                               )
        gconv = [enc]
        #encoder/decoder layer
        if recurrent:
            encdec = Deep_GATE_Conv(node_in_channels = hidden_channels,
                                    node_out_channels = hidden_channels,
                                    edge_in_channels = edgechannels,
                                    edge_out_channels = edgechannels,
                                    heads = heads,
                                    dropout = fc_dropout,
                                    attention_channels = attention_channels,
                                    parameter_efficient = parameter_efficient,
                                    principal_neighbourhood_aggregation = principal_neighbourhood_aggregation,
                                    deg = deg,
                                    aggr=aggr
                                    )
            for idx in np.arange(num_graph_convs-1):
                gconv.append(encdec)
        else:
            for idx in np.arange(num_graph_convs-1):
                enc = Deep_GATE_Conv(node_in_channels = hidden_channels,
                                     node_out_channels = hidden_channels,
                                     edge_in_channels = edgechannels,
                                     edge_out_channels = edgechannels,
                                     heads = heads,
                                     dropout = fc_dropout,
                                     attention_channels = attention_channels,
                                     parameter_efficient = parameter_efficient,
                                     principal_neighbourhood_aggregation = principal_neighbourhood_aggregation,
                                     deg = deg,
                                     aggr=aggr
                                    )
                gconv.append(enc)

        self.gconv = Sequential(*gconv)

        #fully connected channels
        fc_channels = [fc_channels]*num_fc 
        fc_channels = [hidden_channels]+fc_channels
        lin = []
        for idx in torch.arange(num_fc):
            lin.append(BatchNorm1d(fc_channels[idx]))
            lin.append(torch.nn.ReLU())
            lin.append(torch.nn.Dropout(p=fc_dropout))
            lin.append(Linear(fc_channels[idx],fc_channels[idx+1]))
        self.lin = Sequential(*lin)

    def forward(self,
                batch,
                propagate_messages = True
               ):
        batch.edge_attr[torch.isnan(batch.edge_attr)] = 0
        batch.x[torch.isnan(batch.x)] = 0
        
        #randomly drop edges
        batch.edge_index, batch.edge_attr = dropout_adj(batch.edge_index,
                                                         edge_attr=batch.edge_attr,
                                                         p=self.edge_dropout)
        
        #initial dropout and embedding
        batch.x = self.embedding[0](batch.x.float())
        for mod in self.embedding[1:]:               
            batch.x = batch.x + mod(batch.x.float())

        #positional encoding
        if self.positional_encoding and propagate_messages:
            batch.x = self.posencoder(batch.x,
                                      batch.batch)
        
        batch.propagate_messages = propagate_messages
        #graph convolutions
        batch = self.gconv(batch)

        #fully connected linear layers
        x = self.lin(batch.x)
        
        return x

'''
Adds in separate encoding for local promoter info
'''
class GATE_promoter_module(torch.nn.Module):
    def __init__(self,
                 hidden_channels=20,
                 inchannels = 15,
                 edgechannels = 2,
                 heads = 5,
                 num_graph_convs = 2,
                 embedding_layers = 2,
                 num_fc = 1,
                 fc_channels = 15,
                 positional_encoding = True,
                 pos_embedding_dropout = 0.1,
                 fc_dropout = 0.5,
                 edge_dropout = 0.1,
                 outchannels = 3,
                 recurrent = True,
                 numnodespergraph = NUMNODESPERGRAPH,
                 parameter_efficient = True,
                 attention_channels = None,
                 principal_neighbourhood_aggregation = False,
                 deg = None,
                 aggr = 'add'
                ):
        super().__init__()
        self.numnodespergraph = numnodespergraph
        self.inchannels = inchannels
        
        self.module = GATE_module(hidden_channels,
                                  inchannels,
                                  edgechannels,
                                  heads,
                                  num_graph_convs,
                                  embedding_layers,
                                  num_fc,
                                  fc_channels,
                                  positional_encoding,
                                  pos_embedding_dropout,
                                  fc_dropout,
                                  edge_dropout,
                                  recurrent,
                                  parameter_efficient,
                                  attention_channels,
                                  principal_neighbourhood_aggregation,
                                  deg,
                                  aggr
                                 )
        
        #final readout function
        self.readout = Linear(fc_channels, outchannels)
        
        #weighted attention
        self.local_weight_att = Linear(2*fc_channels,1)                
        
    def forward(self,
                batch,
                use_prom = True,
                use_graph = True,
                return_local_weight = False,
                return_embedding = False
               ):
        x = self.module(batch,
                        propagate_messages = use_graph
                       )
        
        if not use_prom:
            #extracting node of interest from graph
            x = get_middle_features(x,
                                    numnodes = self.numnodespergraph)
            x = self.readout(x)
            if return_local_weight:
                return x, 0
            else:
                return x
        
        prom_x = batch.prom_x.view(-1,self.inchannels).float()
        prom_x[torch.isnan(prom_x)] = 0
        batch.x = prom_x
        prom_x = self.module(batch, 
                             propagate_messages = False)
        
        #extracting node of interest from graph
        x = get_middle_features(x,
                                numnodes = self.numnodespergraph)
        
        #combining node-level and promoter level representations via an attention mechanism
        att_in = F.leaky_relu(torch.cat([prom_x,
                                         x], 
                                        dim = 1))
        
        local_weight = sigmoid(self.local_weight_att(att_in)) 
        
        x = local_weight*prom_x + (1-local_weight)*x
        
        if return_embedding:
            return x
        # 4. Apply readout layers
        x = self.readout(x)
        
        if return_local_weight:
            return x, local_weight
        else:
            return x

'''
Get a readout from a GATE module for the mean and logstd of a latent variable distribution
'''
class GATE_variational_encoder(torch.nn.Module):
    def __init__(self,
                 hidden_channels=20,
                 inchannels = 15,
                 edgechannels=1,
                 heads = 4,
                 num_graph_convs = 2,
                 embedding_layers = 2,
                 num_fc = 1,
                 fc_channels = 15,
                 positional_encoding = True,
                 pos_embedding_dropout = 0.1,
                 fc_dropout = 0.5,
                 edge_dropout = 0.1,
                 outchannels = 2,
                 recurrent = True,
                 parameter_efficient = False,
                 attention_channels = None,
                 principal_neighbourhood_aggregation = False,
                 deg = None,
                 aggr = 'add'
                ):
        super().__init__()
        self.module = GATE_module(hidden_channels,
                                  inchannels,
                                  edgechannels,
                                  heads,
                                  num_graph_convs,
                                  embedding_layers,
                                  num_fc,
                                  fc_channels,
                                  positional_encoding,
                                  pos_embedding_dropout,
                                  fc_dropout,
                                  edge_dropout,
                                  recurrent,
                                  parameter_efficient,
                                  attention_channels,
                                  principal_neighbourhood_aggregation,
                                  deg,
                                  aggr
                                 )
        
        #final readout function
        self.readout_mu = Linear(fc_channels, 
                                 outchannels)
        self.readout_logstd = Linear(fc_channels, 
                                     outchannels)                
        
    def forward(self,
                batch):
        x = self.module(batch)

        #Apply readout layers
        mu = self.readout_mu(x)
        logstd = self.readout_logstd(x)

        return mu, logstd

'''
Decode using a GATE module
'''
class GATE_decoder(torch.nn.Module):
    def __init__(self,
                 hidden_channels=20,
                 inchannels = 2,
                 edgechannels = 1,
                 heads = 4,
                 num_graph_convs = 2,
                 embedding_layers = 2,
                 num_fc = 1,
                 fc_channels = 15,
                 positional_encoding = True,
                 pos_embedding_dropout = 0.1,
                 fc_dropout = 0.5,
                 edge_dropout = 0.1,
                 outchannels = 15,
                 recurrent = True,
                 parameter_efficient = True,
                 attention_channels = None,
                 principal_neighbourhood_aggregation = False,
                 deg = None,
                 aggr = 'add'
                ):
        super().__init__()
        self.module = GATE_module(hidden_channels,
                                  inchannels,
                                  edgechannels,
                                  heads,
                                  num_graph_convs,
                                  embedding_layers,
                                  num_fc,
                                  fc_channels,
                                  positional_encoding,
                                  pos_embedding_dropout,
                                  fc_dropout,
                                  edge_dropout,
                                  recurrent,
                                  parameter_efficient,
                                  attention_channels,
                                  principal_neighbourhood_aggregation,
                                  deg,
                                  aggr
                                 )
        
        #final readout function
        self.readout = Linear(fc_channels, 
                              outchannels)
        
    def forward(self,
                batch):
        x = self.module(batch)

        #Apply readout layers
        return self.readout(x)