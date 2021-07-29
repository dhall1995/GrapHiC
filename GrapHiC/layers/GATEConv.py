from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor,PairTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import GRUCell as GRUCell
from torch.utils.checkpoint import checkpoint

from torch.nn import Parameter, Linear, Dropout
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import MessageNorm,BatchNorm
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, dropout_adj

from torch_geometric.nn.inits import glorot, zeros
from .utils import GRU

def stats(arr):
    print(f"\tmin:\t{torch.min(arr).item()}")
    print(f"\tmax:\t{torch.max(arr).item()}")
    print(f"\tmean:\t{torch.mean(arr).item()}")
    
class Static_GATEConv(MessagePassing):
    r"""The graph attentional operator from the `"Graph Attention Networks"
    <https://arxiv.org/abs/1710.10903>`_ paper
    .. math::
        \mathbf{x}^{\prime}_i = \alpha_{i,i}\mathbf{\Theta}\mathbf{x}_{i} +
        \sum_{j \in \mathcal{N}(i)} \alpha_{i,j}\mathbf{\Theta}\mathbf{x}_{j},
    where the attention coefficients :math:`\alpha_{i,j}` are computed as
    .. math::
        \alpha_{i,j} =
        \frac{
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_j]
        \right)\right)}
        {\sum_{k \in \mathcal{N}(i) \cup \{ i \}}
        \exp\left(\mathrm{LeakyReLU}\left(\mathbf{a}^{\top}
        [\mathbf{\Theta}\mathbf{x}_i \, \Vert \, \mathbf{\Theta}\mathbf{x}_k]
        \right)\right)}.
    Args:
        in_channels (int or tuple): Size of each input sample. A tuple
            corresponds to the sizes of source and target dimensionalities.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, 
                 node_in_channels: Union[int, Tuple[int, int]],
                 node_out_channels: int,
                 edge_in_channels: int,
                 edge_out_channels: int,
                 heads: int = 1, 
                 concat: bool = True,
                 negative_slope: float = 0.2, 
                 dropout: float = 0.0,
                 add_self_loops: bool = True, 
                 node_bias: bool = True,
                 edge_bias: bool = True,
                 **kwargs):
        kwargs.setdefault('aggr', 'add')
        super(GATEConv, self).__init__(node_dim=0, **kwargs)

        self.node_in_channels = node_in_channels
        self.node_out_channels = node_out_channels
        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if isinstance(in_channels, int):
            self.lin_l = Linear(in_channels, heads * node_out_channels, bias=False)
            self.lin_r = self.lin_l
        else:
            self.lin_l = Linear(in_channels[0], heads * node_out_channels, False)
            self.lin_r = Linear(in_channels[1], heads * node_out_channels, False)
        
        self.lin_e = Linear(edge_channels, heads*edge_out_channels, bias=False)

        self.att_l = Parameter(torch.Tensor(1, heads, node_out_channels))
        self.att_r = Parameter(torch.Tensor(1, heads, node_out_channels))
        self.att_e = Parameter(torch.Tensor(1, heads, edge_out_channels))

        if node_bias and concat:
            self.node_bias = Parameter(torch.Tensor(heads * node_out_channels))
        elif node_bias and not concat:
            self.node_bias = Parameter(torch.Tensor(node_out_channels))
        else:
            self.register_parameter('node_bias', None)
            
        if edge_bias and concat:
            self.edge_bias = Parameter(torch.Tensor(heads * edge_out_channels))
        elif edge_bias and not concat:
            self.edge_bias = Parameter(torch.Tensor(edge_out_channels))
        else:
            self.register_parameter('edge_bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.lin_e.weight)
        glorot(self.att_l)
        glorot(self.att_r)
        glorot(self.att_e)
        zeros(self.node_bias)
        zeros(self.edge_bias)

    def forward(self, 
                x: Union[Tensor, OptPairTensor],
                edge_attr: Tensor, 
                edge_index: Adj,
                size: Size = None, 
                return_attention_weights=None):
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, OptPairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, OptPairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """        
        H, C, E = self.heads, self.node_out_channels, self.edge_out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        
        if isinstance(x, Tensor):     
            assert x.dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = x_r = self.lin_l(x).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            alpha_r = (x_r * self.att_r).sum(dim=-1)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2, 'Static graphs not supported in `GATConv`.'
            x_l = self.lin_l(x_l).view(-1, H, C)
            alpha_l = (x_l * self.att_l).sum(dim=-1)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)
                alpha_r = (x_r * self.att_r).sum(dim=-1)
         
        edge_attr = self.lin_e(edge_attr).view(-1,H,E)
        alpha_e = (edge_attr * self.att_e).sum(dim=-1)
        
        assert x_l is not None
        assert alpha_l is not None
        assert alpha_e is not None

        # propagate_type: (x: OptPairTensor, alpha: OptPairTensor)
        out = self.propagate(edge_index, 
                             x=(x_l, x_r),
                             alpha=(alpha_l, 
                                    alpha_r),
                             alpha_e = alpha_e,
                             size=size)

        alpha = self._alpha
        self._alpha = None
        
        if self.concat:
            out = out.view(-1, self.heads * self.node_out_channels)
            edge_attr = out.view(-1, self.heads * self.edge_out_channels)
        else:
            out = out.max(dim=1)[0]
            edge_attr = edge_attr.max(dim=1)[0]

        if self.node_bias is not None:
            out += self.node_bias
        
        if self.edge_bias is not None:
            edge_attr += self.edge_bias
        
        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, edge_attr, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_attr, edge_index.set_value(alpha, layout='coo')
        else:
            return out, edge_attr

    def message(self, 
                x_j: Tensor, 
                alpha_j: Tensor, 
                alpha_i: OptTensor,
                alpha_e: Tensor,
                index: Tensor, 
                ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        alpha = alpha_j + alpha_e if alpha_i is None else alpha_j + alpha_i + alpha_e
        alpha = F.leaky_relu(alpha, self.negative_slope)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}(node linear in:{}, node linear out:{}, edge linear in: {}, edge linear out: {}, heads={})'.format(self.__class__.__name__,
                                             self.node_in_channels,
                                             self.node_out_channels,
                                             self.edge_in_channels,
                                             self.edge_out_channels,
                                             self.heads)
    


    
'''
Module to perform deep edge weighted GAT layers with pre-activation skip-connections 
'''
class Static_Deep_GATE_Conv(torch.nn.Module):
    def __init__(self,
                 node_inchannels,
                 node_outchannels,
                 edge_inchannels,
                 edge_outchannels,
                 heads = 4,
                 node_dropout = 0.1,
                 edge_dropout = 0.1
                ):
        super().__init__()
        self.conv = Static_GATEConv(node_in_channels = node_inchannels, 
                                    node_out_channels = node_outchannels,
                                    edge_in_channels = edge_inchannels,
                                    edge_out_channels = edge_outchannels,
                                    heads = heads,
                                    concat = False
                                   )
        self.node_norm = BatchNorm(node_inchannels)
        self.edge_norm = BatchNorm(edge_inchannels)
        self.node_aggr = MessageNorm(learn_scale = True)
        self.edge_aggr = MessageNorm(learn_scale = True)
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
        self.edge_channels = edge_inchannels
    
    def reset_parameters(self):
        self.conv.reset_parameters()
        self.node_norm.reset_parameters()
        self.edge_norm.reset_parameters()


    def forward(self, batch):
        """"""
        #print("############################LAYER########################")
        #print("INPUT FEATURES")
        #print("nodes:")
        #stats(batch.x)
        #print("edges:")
        #stats(batch.edge_attr)
        
        h = self.node_norm(batch.x.float())
        h = F.relu(h)
        h = F.dropout(h, p=self.node_dropout)
        
        edge_h = self.edge_norm(batch.edge_attr.float())
        edge_h = F.relu(edge_h)
        batch.edge_index, temp_edge_attrs = dropout_adj(batch.edge_index, 
                                                        edge_attr=torch.cat([edge_h,batch.edge_attr],
                                                                            dim=1),
                                                        p=self.edge_dropout)
        edge_h, batch.edge_attr = temp_edge_attrs[:,:self.edge_channels],temp_edge_attrs[:,self.edge_channels:]
        
        h, edge_h = self.conv(h.float(),
                              edge_h.float(),
                              batch.edge_index
                             )
        
        #print("MESSAGES")
        #print("nodes:")
        #stats(h)
        #print("edges:")
        #stats(edge_h)
        
        
        batch.x = self.node_aggr(batch.x, h)
        batch.edge_attr = self.edge_aggr(batch.edge_attr, edge_h)
        
        #print("OUTPUT FEATURES")
        #print("nodes:")
        #stats(batch.x)
        #print("edges:")      
        #stats(batch.edge_attr)
        
        return batch

class GATEConv(MessagePassing):
    r"""Args:
        in_channels (int): Size of each input sample.
        out_channels (int): Size of each output sample.
        heads (int, optional): Number of multi-head-attentions.
            (default: :obj:`1`)
        concat (bool, optional): If set to :obj:`False`, the multi-head
            attentions are averaged instead of concatenated.
            (default: :obj:`True`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
        bias (bool, optional): If set to :obj:`False`, the layer will not learn
            an additive bias. (default: :obj:`True`)
        share_weights (bool, optional): If set to :obj:`True`, the same matrix
            will be applied to the source and the target node of every edge.
            (default: :obj:`False`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.
    """
    _alpha: OptTensor

    def __init__(self, 
                 node_in_channels: int,
                 node_out_channels: int,
                 edge_in_channels: int,
                 edge_out_channels: int,
                 heads: int = 1, 
                 concat: bool = True,
                 negative_slope: float = 0.2, 
                 dropout: float = 0.,
                 attention_channels: int = None,
                 node_bias: bool = True,
                 edge_bias: bool = True,
                 att_bias: bool = True,
                 share_weights: bool = True,
                 **kwargs):
        super(GATEConv, self).__init__(node_dim=0, **kwargs)

        self.node_in_channels = node_in_channels
        self.node_out_channels = node_out_channels
        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels
        if attention_channels is None:
            self.attention_channels = node_out_channels
        else:
            self.attention_channels = attention_channels
        self.heads = heads
        self.concat = concat
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.share_weights = share_weights

        self.lin_l = Linear(node_in_channels, heads * node_out_channels, bias=node_bias)
        if share_weights:
            self.lin_r = self.lin_l
        else:
            self.lin_r = Linear(node_in_channels, heads * node_out_channels, bias=node_bias)
        self.lin_e = Linear(edge_in_channels, heads*edge_out_channels, bias=edge_bias)
        
        self.lin_att = Linear((2*node_out_channels)+(edge_out_channels), 
                              self.attention_channels, 
                              bias=att_bias)
        self.att = Parameter(torch.Tensor(1, heads, self.attention_channels))
        
        if concat:
            self.node_lin_out = Linear(heads*node_out_channels,
                                       node_out_channels,
                                       bias=node_bias)
            self.edge_lin_out = Linear(heads*edge_out_channels,
                                       edge_out_channels,
                                       bias=edge_bias)


        if node_bias and not concat:
            self.node_bias = Parameter(torch.Tensor(node_out_channels))
        else:
            self.register_parameter('node_bias', None)
            
        if edge_bias and not concat:
            self.edge_bias = Parameter(torch.Tensor(edge_out_channels))
        else:
            self.register_parameter('edge_bias', None)

        self._alpha = None

        self.reset_parameters()

    def reset_parameters(self):
        glorot(self.lin_l.weight)
        glorot(self.lin_r.weight)
        glorot(self.lin_e.weight)
        glorot(self.att)
        glorot(self.lin_att.weight)
        if self.concat:
            glorot(self.node_lin_out.weight)
            glorot(self.edge_lin_out.weight)
         
        zeros(self.node_bias)
        zeros(self.edge_bias)


    def forward(self, 
                x: Union[Tensor, PairTensor], 
                edge_attr: Tensor,
                edge_index: Adj,
                size: Size = None, 
                return_attention_weights: bool = None):
        # type: (Union[Tensor, PairTensor], Tensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, Size, NoneType) -> Tensor  # noqa
        # type: (Union[Tensor, PairTensor], Tensor, Size, bool) -> Tuple[Tensor, Tuple[Tensor, Tensor]]  # noqa
        # type: (Union[Tensor, PairTensor], SparseTensor, Size, bool) -> Tuple[Tensor, SparseTensor]  # noqa
        r"""
        Args:
            return_attention_weights (bool, optional): If set to :obj:`True`,
                will additionally return the tuple
                :obj:`(edge_index, attention_weights)`, holding the computed
                attention weights for each edge. (default: :obj:`None`)
        """
        H, C, E = self.heads, self.node_out_channels, self.edge_out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        if isinstance(x, Tensor):
            assert x.dim() == 2
            x_l = self.lin_l(x).view(-1, H, C)
            if self.share_weights:
                x_r = x_l
            else:
                x_r = self.lin_r(x).view(-1, H, C)
        else:
            x_l, x_r = x[0], x[1]
            assert x[0].dim() == 2
            x_l = self.lin_l(x_l).view(-1, H, C)
            if x_r is not None:
                x_r = self.lin_r(x_r).view(-1, H, C)

        assert x_l is not None
        assert x_r is not None
        
        edge_attr = self.lin_e(edge_attr).view(-1,H,E)

        # propagate_type: (x: PairTensor)
        out = self.propagate(edge_index, 
                             x=(x_l, x_r),
                             edge_ij = edge_attr,
                             size=size)

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, self.heads * self.node_out_channels)
            out = self.node_lin_out(out)
            
            edge_attr = edge_attr.view(-1, self.heads*self.edge_out_channels)
            edge_attr = self.edge_lin_out(edge_attr)
        else:
            out = out.mean(dim=1)
            edge_attr = edge_attr.mean(dim=1)

            if self.node_bias is not None:
                out += self.node_bias
            
            if self.edge_bias is not None:
                out += self.edge_bias

        if isinstance(return_attention_weights, bool):
            assert alpha is not None
            if isinstance(edge_index, Tensor):
                return out, edge_attr, (edge_index, alpha)
            elif isinstance(edge_index, SparseTensor):
                return out, edge_attr, edge_index.set_value(alpha, layout='coo')
        else:
            return out, edge_attr


    def message(self, 
                x_j: Tensor, 
                x_i: Tensor,
                edge_ij: Tensor,
                index: Tensor, ptr: OptTensor,
                size_i: Optional[int]) -> Tensor:
        x = self.lin_att(torch.cat([x_i,x_j,edge_ij],
                                   axis = -1)
                        )
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)

    def __repr__(self):
        return '{}(node linear in:{}, node linear out:{}, edge linear in: {}, edge linear out: {}, heads={}, attention channels {})'.format(self.__class__.__name__,
                                             self.node_in_channels,
                                             self.node_out_channels,
                                             self.edge_in_channels,
                                             self.edge_out_channels,
                                             self.heads,
                                             self.attention_channels)
    

    

class Deep_GATE_Conv(torch.nn.Module):
    def __init__(self,
                 node_in_channels,
                 node_out_channels,
                 edge_in_channels,
                 edge_out_channels,
                 heads = 4,
                 node_dropout = 0.1,
                 edge_dropout = 0.1
                ):
        super().__init__()
        self.conv = GATEConv(node_in_channels = node_in_channels, 
                               node_out_channels = node_out_channels,
                               edge_in_channels = edge_in_channels,
                               edge_out_channels = edge_out_channels,
                               heads = heads,
                               concat = True
                              )
        self.node_norm = BatchNorm(node_in_channels)
        self.edge_norm = BatchNorm(edge_in_channels)
        #self.node_aggr = MessageNorm(learn_scale = True)
        #self.edge_aggr = MessageNorm(learn_scale = True)
        self.node_aggr = GRUCell(node_in_channels,node_out_channels)
        self.edge_aggr = GRUCell(edge_in_channels,edge_out_channels)
        self.node_dropout = node_dropout
        self.edge_dropout = edge_dropout
        self.edge_channels = edge_in_channels
    
    def reset_parameters(self):
        self.conv.reset_parameters()
        self.node_norm.reset_parameters()
        self.edge_norm.reset_parameters()


    def forward(self, batch):
        """"""
        h = self.node_norm(batch.x.float())
        h = F.relu(h)
        h = F.dropout(h, p=self.node_dropout)
        
        edge_h = self.edge_norm(batch.edge_attr.float())
        edge_h = F.relu(edge_h)
        batch.edge_index, temp_edge_attrs = dropout_adj(batch.edge_index, 
                                                        edge_attr=torch.cat([edge_h,batch.edge_attr],
                                                                            dim=1),
                                                        p=self.edge_dropout)
        edge_h, batch.edge_attr = temp_edge_attrs[:,:self.edge_channels],temp_edge_attrs[:,self.edge_channels:]
        
        h, edge_h = self.conv(h.float(),
                              edge_h.float(),
                              batch.edge_index
                             )
        
        batch.x = self.node_aggr(batch.x, 
                                 h
                                )
        
        batch.edge_attr = self.edge_aggr(batch.edge_attr.float(), 
                                         edge_h
                                        )
        
        return batch
