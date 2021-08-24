from typing import Union, Tuple, Optional, List
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor,PairTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import GRUCell as GRUCell
from torch.utils.checkpoint import checkpoint

from torch.nn import Parameter, Linear, Dropout
from torch_sparse import SparseTensor, set_diag
from torch_scatter import scatter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.norm import MessageNorm,BatchNorm
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, dropout_adj,degree

from torch_geometric.nn.inits import glorot, zeros
from .utils import GRU

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
                 parameter_efficient: bool = True,
                 principal_neighbourhood_aggregation: bool = False,
                 aggregators: List[str]= ['mean', 'min', 'max', 'std'], 
                 scalers: List[str]=['identity', 'amplification', 'attenuation'],
                 deg = None,
                 **kwargs):
        super(GATEConv, self).__init__(node_dim=0, **kwargs)
        
        self.pna = principal_neighbourhood_aggregation
        if self.pna:
            deg = deg.to(torch.float)
            self.avg_deg: Dict[str, float] = {
                'lin': deg.mean().item(),
                'log': (deg + 1).log().mean().item(),
                'exp': deg.exp().mean().item(),
            }
        self.aggregators = aggregators
        self.scalers = scalers
        self.node_in_channels = node_in_channels
        self.node_out_channels = node_out_channels
        self.edge_in_channels = edge_in_channels
        self.edge_out_channels = edge_out_channels
        
        if not parameter_efficient and attention_channels is not None:
            self.attention_channels = attention_channels
        else:
            self.attention_channels = (2*node_out_channels)+(edge_out_channels)
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
        
        if not parameter_efficient:
            self.lin_att = Linear((2*node_out_channels)+(edge_out_channels), 
                                  self.attention_channels, 
                                  bias=att_bias)
        else:
            self.lin_att = None
        
        self.att = Parameter(torch.Tensor(1, heads, self.attention_channels))
        
        if self.pna:
            self.node_pna_channels = (len(aggregators) * len(scalers)) * node_out_channels
        else:
            self.node_pna_channels = node_out_channels
            
        if concat:
            self.node_lin_out = Linear(heads*self.node_pna_channels,
                                       node_out_channels,
                                       bias=node_bias)
            self.edge_lin_out = Linear(heads*edge_out_channels,
                                       edge_out_channels,
                                       bias=edge_bias)
        elif self.pna:
            self.node_lin_out = Linear(self.node_pna_channels,
                                       node_out_channels,
                                       bias=node_bias)


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
        if self.lin_att is not None:
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
                return_attention_weights: bool = None,
                propagate_messages: bool = True
               ):
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

        if propagate_messages:
            # propagate_type: (x: PairTensor)
            out = self.propagate(edge_index,
                                 x=(x_l, x_r),
                                 edge_ij = edge_attr,
                                 size=size)
        else:
            out = x_l

        alpha = self._alpha
        self._alpha = None

        if self.concat:
            if self.pna and not propagate_messages:
                out = out.repeat(1, 1, (len(self.aggregators) * len(self.scalers)))
            out = out.view(-1,
                           self.heads * self.node_pna_channels)
            out = self.node_lin_out(out)
            
            edge_attr = edge_attr.view(-1, 
                                       self.heads * self.edge_out_channels)
            edge_attr = self.edge_lin_out(edge_attr)
        else:
            out = out.mean(dim=1)
            edge_attr = edge_attr.mean(dim=1)

            if self.node_bias is not None:
                out += self.node_bias
            
            if self.edge_bias is not None:
                out += self.edge_bias
            
            if self.pna:
                out = self.node_lin_out(out)

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
        x = torch.cat([x_i,x_j,edge_ij],
                      axis = -1)
        if self.lin_att is not None:
            x = self.lin_att(x)
        x = F.leaky_relu(x, self.negative_slope)
        alpha = (x * self.att).sum(dim=-1)
        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)
        return x_j * alpha.unsqueeze(-1)
    
    def aggregate(self, 
                  inputs: Tensor, 
                  index: Tensor,
                  dim_size: Optional[int] = None) -> Tensor:
        if self.pna:
            outs = []
            for aggregator in self.aggregators:
                if aggregator == 'sum':
                    out = scatter(inputs, index, 0, None, dim_size, reduce='sum')
                elif aggregator == 'mean':
                    out = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                elif aggregator == 'min':
                    out = scatter(inputs, index, 0, None, dim_size, reduce='min')
                elif aggregator == 'max':
                    out = scatter(inputs, index, 0, None, dim_size, reduce='max')
                elif aggregator == 'var' or aggregator == 'std':
                    mean = scatter(inputs, index, 0, None, dim_size, reduce='mean')
                    mean_squares = scatter(inputs * inputs, index, 0, None,
                                       dim_size, reduce='mean')
                    out = mean_squares - mean * mean
                    if aggregator == 'std':
                        out = torch.sqrt(torch.relu(out) + 1e-5)
                else:
                    raise ValueError(f'Unknown aggregator "{aggregator}".')
                outs.append(out)
            out = torch.cat(outs, dim=-1)

            deg = degree(index, dim_size, dtype=inputs.dtype)
            deg = deg.clamp_(1).view(-1, 1, 1)

            outs = []
            for scaler in self.scalers:
                if scaler == 'identity':
                    pass
                elif scaler == 'amplification':
                    out = out * (torch.log(deg + 1) / self.avg_deg['log'])
                elif scaler == 'attenuation':
                    out = out * (self.avg_deg['log'] / torch.log(deg + 1))
                elif scaler == 'linear':
                    out = out * (deg / self.avg_deg['lin'])
                elif scaler == 'inverse_linear':
                    out = out * (self.avg_deg['lin'] / deg)
                else:
                    raise ValueError(f'Unknown scaler "{scaler}".')
                outs.append(out)
            return torch.cat(outs, dim=-1)
        else:
            return scatter(inputs, 
                           index, 
                           dim=self.node_dim, 
                           dim_size=dim_size,
                           reduce=self.aggr)

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
                 heads = 5,
                 dropout = 0.1,
                 parameter_efficient = True,
                 attention_channels = None,
                 principal_neighbourhood_aggregation = False,
                 deg = None,
                 aggr = 'add'
                ):
        super().__init__()
        self.conv = GATEConv(node_in_channels = node_in_channels,
                             node_out_channels = node_out_channels,
                             edge_in_channels = edge_in_channels,
                             edge_out_channels = edge_out_channels,
                             heads = heads,
                             concat = True,
                             attention_channels = attention_channels,
                             parameter_efficient = parameter_efficient,
                             principal_neighbourhood_aggregation = principal_neighbourhood_aggregation,
                             deg = deg,
                             aggr = aggr
                              )
        self.node_norm = BatchNorm(node_in_channels)
        self.edge_norm = BatchNorm(edge_in_channels)
        self.node_aggr = GRUCell(node_in_channels,node_out_channels)
        self.edge_aggr = GRUCell(edge_in_channels,edge_out_channels)
        self.dropout = dropout
        self.edge_channels = edge_in_channels
    
    def reset_parameters(self):
        self.conv.reset_parameters()
        self.node_norm.reset_parameters()
        self.edge_norm.reset_parameters()


    def forward(self, 
                batch):
        """"""
        h = self.node_norm(batch.x.float())
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout)
        
        edge_h = self.edge_norm(batch.edge_attr.float())
        edge_h = F.relu(edge_h)
        edge_h = F.dropout(edge_h, p=self.dropout)
        
        if batch.propagate_messages is None:
            batch.propagate_messages = True
        h, edge_h = self.conv(h.float(),
                              edge_h.float(),
                              batch.edge_index,
                              propagate_messages = batch.propagate_messages
                             )
        
        batch.x = self.node_aggr(batch.x, 
                                 h
                                )
        
        batch.edge_attr = self.edge_aggr(batch.edge_attr.float(), 
                                         edge_h
                                        )
        
        return batch
