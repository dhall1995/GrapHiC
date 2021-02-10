from scipy.interpolate import CubicSpline
import torch
import numpy as np
import h5py as h
import pandas as pd
import os.path as osp

def make_basic_feature_matrix(chrlims, binSize, include_basepairs= False):
    '''
    Utility to create a bare bones feature matrix given a contact map. If we dont enode information
    such as contact length within the edges then this takes into account the fact that, no matter what
    features we choose to include or which experimental consition we have, each bin will have a fixed
    chromosome and position along the chromosome. 
    
    Arguments:
    
    - chrlims: Dictionary detailing the start and end basepairs of each chromosome in the contact dictionary.
               NOTE: the chromosome limits are inclusive i.e. for each CHR_A we should have
               chromo_limits[CHR_A] = (start_A,end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - binSize: The length (in basepairs) of each chromatin bin in our contact map.
    
    Returns:
    
    - totalfmat: An (N,21) shape tensor where index 21 details the basepair of the bin while the first
                 20 indexes are one-hot encodings of the chromosomes.
    '''
    order = [str(i+1) for i in np.arange(19)] + ['X']
    
    if include_basepairs:
        totalfmat = np.empty((0,21))
    else:
        totalfmat = np.empty((0,20))
    for chrom in order:
        onehot = one_hot(chrom).astype(int)
        
        lenchrom = int((chrlims[chrom][1] - chrlims[chrom][0])/binSize)
        
        fmat = np.tile(onehot, (lenchrom,1))
        if include_basepairs:
            chridxs = np.arange(chrlims[chrom][0], chrlims[chrom][1], binSize).astype(int)
            fmat = np.append(fmat, chridxs[:,None], axis = 1)    
    
        totalfmat = np.append(totalfmat, fmat, axis = 0).astype(int)
        
    totalfmat = torch.tensor(totalfmat, dtype = torch.float)
    
    return totalfmat