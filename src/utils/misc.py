from __future__ import annotations

import logging
from typing import Dict, Optional
import numpy as np

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
    nodes:np.ndarray
)-> np.ndarray:
    """
    Renames the nodes in an edge_index array to be zero-indexed corresponding to some given nodes.
    :param edge_index: edge_index array of size (2, num_edges).
    :param nodes: array specifying the order of the nodes. Must be exhaustive - i.e. there shouldn't appear a value in edge_index which doesn't appear in nodes.
    :return: edge_index zero-indexed 
    """
    node_dict = {node: idx for idx, node in enumerate(nodes)}
    out = np.zeros(edge_index.shape)
    f = lambda x: node_dict[x]
    
    out[0,:] = np.array(list(map(f, edge_index[0,:])))
    out[1,:] = np.array(list(map(f, edge_index[1,:])))
    return out