from typing import Union, Tuple, Optional
from torch_geometric.typing import (OptPairTensor, Adj, Size, NoneType,
                                    OptTensor)

import torch
from torch import Tensor
import torch.nn.functional as F
from torch.nn import Parameter, Linear, Dropout
from torch_sparse import SparseTensor, set_diag
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn import TopKPooling as TKP
from torch_geometric.utils import remove_self_loops, add_self_loops, softmax, dropout_adj

from torch_geometric.nn.inits import glorot, zeros

def stats(arr):
    print(f"\tmin:\t{torch.min(arr).item()}")
    print(f"\tmax:\t{torch.max(arr).item()}")
    print(f"\tmean:\t{torch.mean(arr).item()}")
    
class WEGATConv(MessagePassing):
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
                 in_channels: Union[int, Tuple[int, int]],
                 edge_channels: int,
                 node_out_channels: int,
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
        super(WEGATConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = node_out_channels
        self.edge_channels = edge_channels
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
        H, C, E = self.heads, self.out_channels, self.edge_out_channels

        x_l: OptTensor = None
        x_r: OptTensor = None
        alpha_l: OptTensor = None
        alpha_r: OptTensor = None
        
        #print("Pre-processed node features")
        #stats(x)  
        #print("pre-processed edge features")
        #stats(edge_attr)
        
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
        
        #print("embedded node features")
        #stats(x_l)
        #print("embedded edge features")
        #stats(edge_attr)
        #print("edge attention coefficients")
        #stats(alpha_e)
        #print("node attention coefficients")
        #stats(alpha_l)
        
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
            out = out.view(-1, self.heads * self.out_channels)
            edge_attr = out.view(-1, self.heads * self.edge_out_channels)
        else:
            out = out.max(dim=1)[0]
            edge_attr = edge_attr.max(dim=1)[0]

        if self.node_bias is not None:
            out += self.node_bias
        
        if self.edge_bias is not None:
            edge_attr += self.edge_bias
        
        #print("edge bias")
        #stats(self.edge_bias)
        #print("node bias")
        #stats(self.node_bias)
        #print("propagated out node features (embedded features + bias)")
        #stats(out)
        #print("propagated out edge features (embedded features + bias)")
        #stats(edge_attr)
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
        return '{}({}, {}, heads={})'.format(self.__class__.__name__,
                                             self.in_channels,
                                             self.out_channels, self.heads)

'''
COMBINED 'WEIGHTED EDGE GRAPH ATTENTION' + 'TOP K POOLING LAYERS' 
'''
class WEGAT_TOPK_Conv(torch.nn.Module):
    def __init__(self,
                 node_inchannels,
                 node_outchannels,
                 edge_inchannels,
                 edge_outchannels,
                 heads = 4,
                 dropout = 0.1
                ):
        super().__init__()
        self.conv = WEGATConv(in_channels = node_inchannels, 
                               node_out_channels = node_outchannels,
                               edge_channels = edge_inchannels,
                               edge_out_channels = edge_outchannels,
                               heads = heads,
                               concat = False
                              )
        self.dropout = Dropout(p=dropout)
        self.p = dropout
        
    def forward(self, 
                batch):
        #print("#######################Layer##########################")
        #print("%%%%%%%%%%%%%%%%%%%%%%inputs%%%%%%%%%%%%%%%%%%%%%%%%%%")
        #print("Nodes:")
        #stats(batch.x)
        #print("Edges:")
        #stats(batch.edge_attr)
        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        batch.x, batch.edge_attr = self.conv(batch.x.float(),
                                             batch.edge_attr.float(),
                                             batch.edge_index)
        #print("%%%%%%%%%%%%%%%%%convolved outputs%%%%%%%%%%%%%%%%%%%%")
        #print("%%%%%%%%%%%%%%%%%pre dropout/relu%%%%%%%%%%%%%%%%%%%%%")
        #print("Nodes:")
        #stats(batch.x)
        #print("Edges:")
        #stats(batch.edge_attr)
        #print("%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%")
        batch.x = self.dropout(batch.x)
        batch.x = batch.x.relu()
        
        batch.edge_index, batch.edge_attr = dropout_adj(batch.edge_index, 
                                                        edge_attr=batch.edge_attr, 
                                                        p=self.p)
        batch.edge_attr = batch.edge_attr.relu()
        #print("%%%%%%%%%%%%%%%%%%%%%%outputs%%%%%%%%%%%%%%%%%%%%%%%%%%")
        #print("Nodes:")
        #stats(batch.x)
        #print("Edges:")
        #stats(batch.edge_attr)
        return batch