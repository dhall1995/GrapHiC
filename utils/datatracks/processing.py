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


def interpolate_sc_features(sc_file, feature, chrlims, binSize, shape):
    '''
    Given some input single-cell .nuc file, a feature stored within, and some binning of the genome, this function
    interpolates the single-cell feature from the structural binSize to the input binSize via cubic splines and then
    returns an appropriately binned feature vector for addition to a feature matrix or for downstream analysis. The 
    feature is normalised to [0,1] by default
    
    Arguments:
    
    - sc_file: The single-cell .nuc file containing structural information as well as information regarding the data
               track of interest.
    - feature: The feature name of interest. This assumes an hdf5 file format within the .nuc file of the form:
                        f["dataTracks"]["derived"][{feature}][{chrom}]
               for each {chrom} in our chromosomes of interest and where {feature} is our feature name.
    - chrlims: Dictionary detailing the start and end basepairs of each chromosome in the contact dictionary.
               NOTE: the chromosome limits are inclusive i.e. for each CHR_A we should have
               chromo_limits[CHR_A] = (start_A,end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - binSize: The length (in basepairs) of each chromatin bin in our contact map.
    - shape: Shape of the output feature vector (essentially (N,) where N is the number of nodes).
    
    Returns:
    
    - bdtrack: The interpolated and binned dataTrack given the input binning. Should be a shape (N,) vector.
                                 
    '''
    chroms = [str(i+1) for i in np.arange(19)] + ['X']
    
    bdtrack = -np.ones(shape)
    f = h.File(sc_file,'r')

    mydtrack = f['dataTracks']['derived'][feature]
    mygenpos = {chrom: f['structures']['0']['particles'][chrom]['positions'] for chrom in mydtrack}
    for chrom in mydtrack:
        for idx, pos in enumerate(mygenpos[chrom]):
            try:
                popidx = bp_to_idx(pos + 0.5*binSize, chrom, chrlims, binSize)
                bdtrack[popidx] = np.mean(mydtrack[chrom][:, idx])
            except:
                pass

    f.close()
    
    
    interpolationpoints = {chrom : [idx_to_bp(idx, chrlims, binSize, chroms)[1] + 0.5*binSize for idx in np.where(bdtrack > -1)[0] if idx_to_bp(idx, chrlims, binSize, chroms)[0] == chrom]for chrom in chroms}
    cs = {chrom: CubicSpline(interpolationpoints[chrom],
                         [bdtrack[idx] for idx in np.where(bdtrack > -1)[0] if idx_to_bp(idx, chrlims, binSize, chroms)[0] == chrom]
                        ) for chrom in chroms}

    maxval = np.max(bdtrack[bdtrack > -1])
    minval = np.min(bdtrack[bdtrack > -1])
    for idx in np.arange(len(bdtrack)):
        if bdtrack[idx] == -1:
            chrom, bp = idx_to_bp(idx, chrlims, binSize, chroms)
            bdtrack[idx] = np.maximum(minval, np.minimum(maxval,cs[chrom](bp + 0.5*binSize)))

    return bdtrack


def collate_sc_features(sc_file_list, feature, chrlims, binSize, shape):
    '''
    Given some input list of single-cell Hi-C .nuc files, executes interpolate_sc_features (see above) on each
    input file, saving a large feature matrix of shape (N,K) where K is the number of input cells. Then calculates
    the per node mean and standard deviation of that feature across the given input cells. This is included since
    we may have different number of cells per timepoint but not yet enough cells to form comparable histograms across
    different conditions. With more data it should really become obsolete.
    
    Arguments:
    
    - sc_file_list: List of single-cell .nuc files containing structural information as well as information regarding
                    the data track of interest.
    - feature: The feature name of interest. This assumes an hdf5 file format within the .nuc file of the form:
                        f["dataTracks"]["derived"][{feature}][{chrom}]
               for each {chrom} in our chromosomes of interest and where {feature} is our feature name.
    - chrlims: Dictionary detailing the start and end basepairs of each chromosome in the contact dictionary.
               NOTE: the chromosome limits are inclusive i.e. for each CHR_A we should have
               chromo_limits[CHR_A] = (start_A,end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - binSize: The length (in basepairs) of each chromatin bin in our contact map.
    - shape: Shape of the output feature vector (essentially (N,) where N is the number of nodes).
    
    Returns:
    
    - means: The per-node mean of the feature across the input cells
    - std: The per-node standard deviation of the feature across the input cells
    

    '''
    scfmat = []
    for file in sc_file_list:
        print("Interpolating {} in cell {}".format(feature, file.split("/")[-1]))
        x = interpolate_sc_features(file, feature, chrlims, binSize, shape)[None,:]
        scfmat += [x]
        print(len(scfmat))
        
    scfmat = np.vstack(scfmat)
    
    means = np.mean(scfmat, axis = 0)
    std = np.std(scfmat, axis = 0)
    
    return means, std
