import numpy as np

def one_hot(chrom): 
    '''
    Encodes a chromosome as a one-hot length 20 vector.
    
    Arguments:
    
    - chrom: A chromosome (string). For example, '2'.
    
    Returns: 
    
    - vals: A (1,20) shape array with a 1 in the
            position corresponding to the input
            chromosome
    '''
    vals = np.zeros((1,20))
    
    try:
        c = int(chrom)
        vals[0,c-1] = 1
        return vals
    except:
        if chrom == 'X':
            vals[0,-1] = 1
            return vals
        else:
            raise ValueError
            
def idx_to_bp(idx, chr_lims, binSize, chroms):
    '''
    Utility function to convert from a bin index to basepairs. This
    assumes that chromosomes are concatenated onto each other.
    
    Arguments:
    
    - idx: The index in the concatenated array.
    - chr_lims: Dictionary detailing the start and end basepairs of each chromosome
                in the contact dictionary. NOTE: the chromosome limits are inclusive
                i.e. for each CHR_A we should have chromo_limits[CHR_A] = (start_A,
                end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - binSize: The size of each chromatin bin.
    - chroms: A list of the chromosomes in the order they have been concatenated
              together (usually this will just be ['1', '2',..., '19', 'X']).
    
    Returns:
    - chrom: The chromosome corresponding to the input index.
    - bp: The basepair of the bin on chromosome chrom.
    
    '''
    
    ordering = {idx: chrom for idx, chrom in enumerate(chroms)}
    clens = {idx: int((chr_lims[ordering[idx]][-1] - chr_lims[ordering[idx]][0])/binSize) for idx in ordering}
    tot = 0
    for idx2 in np.arange(len(chroms)):
        tot += clens[idx2]
        if idx <= tot:
            chrom = ordering[idx2]
            chridx = idx - tot + clens[idx2]
            break
    
    bp = chr_lims[chrom][0] + chridx*binSize
    return chrom, bp           
    
        
def bp_to_idx(bp, chrom, chr_lims, binSize):
    '''
    Utility function to convert from a basepair to a bin index if 
    chromosomes are concatenated together. This assumes a 
    concatenation of in the order '1', '2', ..., '19', 'X'.
    
    Arguments:
    - bp: The input basepair.
    - chrom: The input chromosome (should be a string but can deal
             with integers).
    - chr_lims: Dictionary detailing the start and end basepairs of each chromosome
                in the contact dictionary. NOTE: the chromosome limits are inclusive
                i.e. for each CHR_A we should have chromo_limits[CHR_A] = (start_A,
                end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - binSize: The size of each chromatin bin.
    '''
    rounded_bp = binSize*np.floor(bp/binSize)
    chr_idx = int((rounded_bp - chr_lims[chrom][0])/binSize)
    tot = 0
    if chrom != 'X':
        for i in np.arange(1,int(chrom)):
            tot += (chr_lims[str(i)][1] - chr_lims[str(i)][0])/binSize
    else:
        for i in np.arange(1,20):
            tot += (chr_lims[str(i)][1] - chr_lims[str(i)][0])/binSize
    
    return int(tot + chr_idx)