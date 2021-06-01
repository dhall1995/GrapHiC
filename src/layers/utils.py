import torch
from torch.nn import Dropout
import math

class PositionalEncoding(torch.nn.Module):

    def __init__(self, 
                 d_model, 
                 dropout=0.1, 
                 max_len=100,
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