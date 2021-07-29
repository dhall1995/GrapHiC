import torch
from torch.nn import Dropout
import torch.nn.functional as F
import math

class PositionalEncoding(torch.nn.Module):

    def __init__(self, 
                 d_model, 
                 dropout=0.1, 
                 max_len=500,
                 identical_sizes = True
                ):
        super(PositionalEncoding, self).__init__()
        self.identical_sizes = identical_sizes
        self.d_model = d_model
        self.dropout = Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if identical_sizes:
            pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, 
                x, 
                batch):
        if self.identical_sizes:
            return self.forward_identical_sizes(x,
                                           batch)
        else:
            return self.forward_different_sizes(x,
                                           batch)
    
    def forward_different_sizes(self,
                                x,
                                batch):
        num = batch.max()
        for graph in torch.arange(0,num):
            idxs = batch == graph
            x[idxs,:] = x[idxs,:] + self.pe[:torch.sum(idxs), :]
            
        return self.dropout(x)
    
    def forward_identical_sizes(self,
                                x,
                                batch):
        size = torch.sum(batch == 0).item()
        x = x.view(size,-1,self.d_model).float()
        x = x + self.pe[:x.size(0), :]
        x = x.view(-1, self.d_model)
        
        return self.dropout(x)
    
class GRU(torch.nn.Module):
    """
        Wrapper class for the GRU used by the GNN framework, nn.GRU is used for the Gated Recurrent Unit itself
    """

    def __init__(self, 
                 input_size, 
                 hidden_size):
        super(GRU, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = torch.nn.GRU(input_size=input_size, 
                       hidden_size=hidden_size)

    def forward(self, x, y):
        """
        :param x:   shape: (B, N, Din) where Din <= input_size (difference is padded)
        :param y:   shape: (B, N, Dh) where Dh <= hidden_size (difference is padded)
        :return:    shape: (B, N, Dh)
        """
        x = x.unsqueeze(1)
        y = y.unsqueeze(1)
        assert (x.shape[-1] <= self.input_size and y.shape[-1] <= self.hidden_size)

        (B, N,_) = x.shape
        x = x.reshape(1, B * N, -1).contiguous()
        y = y.reshape(1, B * N, -1).contiguous()

        # padding if necessary
        if x.shape[-1] < self.input_size:
            x = F.pad(input=x, pad=[0, self.input_size - x.shape[-1]], mode='constant', value=0)
        if y.shape[-1] < self.hidden_size:
            y = F.pad(input=y, pad=[0, self.hidden_size - y.shape[-1]], mode='constant', value=0)

        x = self.gru(x, y)[1]
        x = x.reshape(B, N, -1)
        return x[:,0,:]
    
def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)