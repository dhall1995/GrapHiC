from __future__ import annotations

import logging
from typing import Dict, Optional,List
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

def ProgressBar(iteration,
                total,
                decimals = 1, 
                length = 100, 
                fill = '█',
                suffix = '',
                prefix = '',
                printEnd = "\r"
               ):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% {suffix}', end = printEnd)
        
        
def buffer_regs(regs: np.ndarray,
                buff: Optional[int] = 1e6,
                lims: Optional[List[int]] = [0,1e9]
               )-> np.ndarray:
    out = np.zeros(regs.shape)
    mean = np.mean(regs,axis = 1).astype('int32')
    out[:,0] = mean - buff
    out[:,1] = mean + buff
    
    out[out<lims[0]] = lims[0]
    out[out>lims[1]] = lims[1]
    
    return out.astype('int32')

def ptg_from_npy(arch):
    out = Data(x = torch.tensor(arch['x'],
                              dtype = torch.double
                              ), 
             edge_index=torch.tensor(
                 arch['edge_index'],
                 dtype=torch.long
                 ),
             edge_attr=torch.tensor(
                 arch['edge_attrs'],
                 dtype = torch.double
                 ),
             y = torch.tensor(
                 arch['target'],
                 dtype = torch.double
                 )
             )
    out.cooler_idxs = torch.tensor(arch['cooler_idxs'],
                                 dtype = torch.long)
    out.name = arch['name']
    out.prom_x = torch.tensor(arch['prom_x'],
                            dtype = torch.double)
    return out

def name_chr(chrom):
    if 'chr' in chrom:
        return str(chrom)
    else:
        return "chr"+str(chrom)


def make_chromo_onehot(
    chromo: str,
    size: int,
    chromosomes: Optional[list] = ["chr{}".format(str(i+1)) for i in np.arange(19)] + ['chrX']
)-> np.array:
    """
    Given a chromosome, casts it as a one hot encoded array of a given size.
    :param chromo: chromosome 
    :param size: Integer specifying how many times to repeat the one-hot encoded vector
    :params chromosomes: List of chromosomes in our total dataset
    :return: one hot array of size (len(chromosomes),size) 
    """
    idx = -1
    for jdx,chromosome in enumerate(chromosomes):
        if chromo == chromosome:
            idx = jdx
            break
    if idx== -1:
        print("Provided chromosome not in list of chromosomes")
        raise
        
    out = np.zeros((size, len(chromosomes)))
    out[:,idx] = 1
    return out

def rename_nodes(
    edge_index: np.ndarray,
    nodes:np.ndarray,
    new_nodes = None
)-> np.ndarray:
    """
    Renames the nodes in an edge_index array to be zero-indexed corresponding to some given nodes.
    :param edge_index: edge_index array of size (2, num_edges).
    :param nodes: array specifying the order of the nodes. Must be exhaustive - i.e. there shouldn't appear a value in edge_index which doesn't appear in nodes.
    :return: edge_index zero-indexed 
    """
    if new_nodes is not None:
        node_dict = {node: new_nodes[idx] for idx, node in enumerate(nodes)}
    else:
        node_dict = {node: idx for idx, node in enumerate(nodes)}
    out = np.zeros(edge_index.shape)
    f = lambda x: node_dict[x]
    
    out[0,:] = np.array(list(map(f, edge_index[0,:])))
    out[1,:] = np.array(list(map(f, edge_index[1,:])))
    return out