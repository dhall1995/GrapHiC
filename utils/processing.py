from scipy.sparse import coo_matrix
from utils.io import load_npz_contacts as lnc
from utils.io import load_data_track, binDataTrack, binDatatrackIDs
from utils.misc import one_hot, bp_to_idx, idx_to_bp


from scipy.interpolate import CubicSpline
import torch
from torch_geometric.utils import is_undirected
import numpy as np
import h5py as h
import pandas as pd
import os.path as osp

from multiprocessing import Pool
from functools import partial


def get_cont_distance(chr_lims, binSize, chroms, const, mycont):
    '''
    Given a some chromosome limits, binSizes and which chromosomes we are interested in, returns the log10 
    contact distance between two indexes, cont. If the two indexes encode bins within different chromosomes 
    then the pair is given a score of -5.
    
    Arguments:
    
    - chr_lims: Dictionary detailing the start and end basepairs of each chromosome in the contact dictionary.
                NOTE: the chromosome limits are inclusive i.e. for each CHR_A we should have
                chromo_limits[CHR_A] = (start_A,end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - binSize: The length (in basepairs) of each chromatin bin in our contact map.
    - chroms: A list of chromosomes in our Hi-C map. 
    - const: Constant to return if indexes in cont correspond to bins on different chromosomes.
    - cont: A 2-tuple of indexes from our contact map. i.e. inputting (i,j) would output the log10 backbone 
            distance between i and j if those indices correspond to bins on the same chromosome or just a
            constant if they correspond to bins on different chromosomes 
         
    
    Returns:
    
    - log10dist: log10 of the backbone distance between those Hi-C indiexes.
    
    
    '''  
    
    idx, cont = mycont[0], mycont[1]
    chrom, chrbp = idx_to_bp(cont[0], chr_lims , binSize, chroms)
    chrom2, chrbp2 = idx_to_bp(cont[1], chr_lims , binSize, chroms)
    
    if (chrom2 == chrom):
        if abs(chrbp2-chrbp) > 0:
            return idx, np.log10(abs(chrbp2-chrbp))
        else:
            return idx, 0
    else:
        return idx, const

def backbone_restraint(row, col, size, index):
    '''
    Given a row and column vectors detailing a cis-Hi-C contact matrix (essentially in in COO format) as well
    as an index, checks whether that index has Hi-C contacts with its adjascent indices along the backbone. 
    
    Arguments:
    
    - row: row indexes from COO format sparse matrix. Assumed to be zero indexed
    - col: column indexes from COO format sparse matrix. Assumed to be zero indexed
    - index: index to be checked     
    
    Returns:
    
    - pos: Bool. If true, then the contact [index, index +1] is in the matrix
    - neg: Bool. If ture, then the contact [index-1, index] is in the matrix
    
    '''  
    
    #Assume backbone restraint satisfied
    pos = True
    neg = True
    
    if index < size-1:
        #if index is at the end of the chr so postive strand restraint already satisfied
        #is index in rows?
        row_idx = row == index
        if np.sum(row_idx) == 0:
            #index not in the rows
            pos = False
        else:
            row_idx_cols = col[row_idx]
            if index+1 not in row_idx_cols:
                pos = False
                
    if index > 0:
        #if index is zero then the negative strand restraint is automatically satisfied
        #Is index in the cols?
        col_idx = col == index
        if np.sum(col_idx) == 0:
            #index not in the cols
            neg = False
        else:
            col_idx_rows = row[col_idx]
            if index-1 not in col_idx_rows:
                neg = False
    
    return pos, neg, index
    
    
    

def make_ptg_pophic_edge_index(file, verbose = True):
    '''
    Creates a host of relevant objects involved in the construction of a pytorch geometric population Hi-C
    edge index object detailing graph edges and strengths. 
    
    Arguments:
    
    - file: .npz file generated using the ncc_bin tool from nuc_tools (see https://github.com/tjs23/nuc_tools)
    - verbose: Bool. Whether to print out progress updates.
    
    Returns:
    
    - shape: resulting shape of the contact matrix
    - binSize: The length (in basepairs) of each chromatin bin in our contact map. 
    - chr_lims: Dictionary detailing the start and end basepairs of each chromosome in the contact dictionary.
                NOTE: the chromosome limits are inclusive i.e. for each CHR_A we should have
                chromo_limits[CHR_A] = (start_A,end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - egde_index: Edge index matrix of the contact map. This is essentially the contact map in COO format
                  where we only detail which nodes are in contact with which other nodes
    - edge_attr: Edge attribute matrix. Here we detail the edge strength of each edge in the conact matrix
    
    
    '''    
    binSize,y,z = lnc(file, store_sparse = True)
    
    
    order = ['chr' + str(i+1) for i in np.arange(19)] + ['chrX']

    offsets = np.cumsum([int((y[i][1]-y[i][0])/binSize) for i in order])
    shape = (offsets[-1], offsets[-1])
    lenchr1 = offsets[0] 
    offsets -= lenchr1
    offsets = {chrom: offsets[i] for i, chrom in enumerate(order)}
    
    myrows = np.empty((1))
    mycols = np.empty((1))
    mydata = np.empty((1))
    for item in z.keys():
        z[item] = z[item].tocoo()
        
        
        chr1 = item[0]
        chr2 = item[1]
        if item[0] != 'chrY' and item[1] != 'chrY':
            rowadd = z[item].row
            coladd = z[item].col
            dataadd = z[item].data
            
            #If same chromosome then check backbone edges
            if item[0] == item[1]:
                backbone_idxs = abs(rowadd - coladd) == 1
                if verbose:
                    print("Checking backbone restraints on chromosome {}".format(item[0]))
                lenchrom = z[item].shape[0]
                if verbose:
                    print("Chromosome length: {}".format(lenchrom))
                fn = partial(backbone_restraint, rowadd[backbone_idxs], coladd[backbone_idxs],lenchrom)
                p = Pool()
                temp_outputs = p.imap(fn, (idx for idx in range(lenchrom)))
                
                postot = 0
                negtot = 0
                for temp_output in temp_outputs:
                    if not temp_output[0]:
                        rowadd = np.append(rowadd, temp_output[2])
                        coladd = np.append(coladd, temp_output[2] + 1)
                        dataadd = np.append(dataadd, 1)
                        postot += 1
                    if not temp_output[1]:
                        rowadd = np.append(rowadd, temp_output[2]-1)
                        coladd = np.append(coladd, temp_output[2])
                        dataadd = np.append(dataadd, 1)
                        negtot += 1
               
                if verbose:
                    print("Added in {} positive strand and {} negative strand backbone contacts for {}".format(postot, negtot, item[0]))
                    print("Done!")
                
            rowadd = rowadd + offsets[chr1]
            coladd = coladd + offsets[chr2]
            
            myrows = np.append(myrows, rowadd).astype(int)
            mycols = np.append(mycols, coladd).astype(int)
            mydata = np.append(mydata, dataadd)
            
            if item[1] != item[0]:
                myrows = np.append(myrows, coladd).astype(int)
                mycols = np.append(mycols, rowadd).astype(int)
                mydata = np.append(mydata, dataadd)
    
    del y['chrY']
    chr_lims = {chrom[3:]: y[chrom] for chrom in y}
    
    temp_coo = coo_matrix((mydata, (myrows, mycols)), shape = shape)
    
    edge_index = np.append(temp_coo.row[None,:], temp_coo.col[None,:], axis = 0)
    edge_attr = temp_coo.data[None,:]

    #add in backbone edges and give each edge an attribute determining whether it is
    #trans or distal cis. This will be the log-length of the edge size for cis contacts
    # and -5 for trans contacts
    chroms = [str(i+1) for i in np.arange(19)] + ['X']
    
    fn = partial(get_cont_distance, chr_lims, binSize, chroms, -5)
    p = Pool()
    lengths = np.zeros(edge_attr.shape)
    
    if verbose:
        print("Adding in edge lengths...")
    temp_outputs = p.imap(fn, ((idx, edge_index[:,idx]) for idx in range(edge_index.shape[1])))
    for temp_output in temp_outputs:
        lengths[0,temp_output[0]] = temp_output[1]
    
    if verbose:
        print("Done!")
    edge_attr = np.append(edge_attr, lengths, axis = 0)

    return  shape, binSize, chr_lims, edge_index, edge_attr



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

def save_sc_feature_to_npz(scfiles, feature, chr_lims, binSize, shape, directory):
    '''
    Basic script to retrieve and save a single-cell feature from a .nuc format file and convert
    to a .npz file for downstream useage.
    
    Arguments:
    
    - scfiles: List of single-cell file paths. These should be .nuc (hdf5) files and should contain the
               following group structure:
                       dataTracks/
                       dataTracks/derived/
    - feature: The feature name of interest. This assumes an hdf5 file format within the .nuc file of the form:
                        f["dataTracks"]["derived"][{feature}][{chrom}]
               for each {chrom} in our chromosomes of interest and where {feature} is our feature name.
    - chrlims: Dictionary detailing the start and end basepairs of each chromosome in the contact dictionary.
               NOTE: the chromosome limits are inclusive i.e. for each CHR_A we should have
               chromo_limits[CHR_A] = (start_A,end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - binSize: The length (in basepairs) of each chromatin bin in our contact map.
    - shape: Shape of the output feature vector (essentially (N,) where N is the number of nodes).
    - directory: the directory in which to save the output .npz file for downstream useage. 
    '''
    chroms = [str(i+1) for i in np.arange(19)] + ['X']
    mean, std = collate_sc_features(scfiles, feature, chr_lims, binSize, shape)
    
    regions = {chrom: [] for chrom in chroms}
    mvals = {chrom: [] for chrom in chroms}
    svals = {chrom: [] for chrom in chroms}
    for idx in np.arange(len(mean)):
        chrom, bp = idx_to_bp(idx, chr_lims, binSize, chroms)
        reg = [bp, bp+binSize]
        mval = [mean[idx]]
        sval = [std[idx]]
        
        regions[chrom].append(reg)
        mvals[chrom].append(mval)
        svals[chrom].append(sval)
    
    m_out_dict = {}
    s_out_dict = {}
    
    for chrom in chroms:
        mrkey = "dataTrack/regions/{}_{}/{}".format('mean', feature, chrom)
        srkey = "dataTrack/regions/{}_{}/{}".format('std', feature, chrom)
        mvkey = "dataTrack/values/{}_{}/{}".format('mean', feature, chrom)
        svkey = "dataTrack/values/{}_{}/{}".format('std', feature, chrom)
            
        m_out_dict[mrkey] = regions[chrom]
        s_out_dict[srkey] = regions[chrom]
        m_out_dict[mvkey] = mvals[chrom]
        s_out_dict[svkey] = svals[chrom]
     
    
    np.savez(osp.join(directory, "{}_mean_{}".format(condition,feature)), **m_out_dict)
    np.savez(osp.join(directory, "{}_std_{}".format(condition, feature)), **s_out_dict)


def get_binned_feature(file, 
                       chrlims,
                       binSize,
                       input_params = None,
                       ID = True):
    '''
    Given some feature of interest (e.g. Nanog ChIP-seq), we want to bin that feature according to some chromosome
    limits and a given binSize. 
    
    Arguments:
    
    - file: .npz file containing keys of the form:
                    "datatrack/{featurename}/regions/{chrom}"
                    "datatrack/{featurename}/values/{chrom}"
            for each {chrom} in our chromosomes of interest and where {featurename} is some descriptive name of
            the feature. In this case the "regions" key value will be an (N_CHROM, 2) shape array where each row
            is some continuous region of the genome in basepairs. The "values" key will be an (N_CHROM,) shape
            array containing values in each of the N_CHROM non-overlapping regions detailed in the "regions" key
            value. 
    - chrlims: Dictionary detailing the start and end basepairs of each chromosome in the contact dictionary.
               NOTE: the chromosome limits are inclusive i.e. for each CHR_A we should have
               chromo_limits[CHR_A] = (start_A,end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - binSize: The length (in basepairs) of each chromatin bin in our contact map.
    - input_params: Dictionary with the keys specifying binning options. You need not specify all keys below but 
                    keys in the dictionary must come from:
                        - "quantitative": Indicates whether the feature in question can be considered as
                                          quantitative or not. ChIPseq experiments without spike-in proteins
                                          produce peak heights which are hard to compare with one another and
                                          therefore we may not want to take into account peak height. In this
                                          case, all peak heights are treated as binary.
                        - "norm_value_by_region": Each region is associated with a value regardless of region
                                                  width. If norm_values_by_region then the value associated
                                                  with each region is multiplied by {region_width}/{binSize}.
                                                  Essentially this means that a 10 basepair region with value
                                                  10 is treated the same as a 100 basepair region with value 1
                                                  if both regions fall entirely within the same bin. Think of
                                                  this as enforcing an 'area under the curve' style binning as
                                                  opposed to weighting all regions equally.
                        - "average": If True, each bin value is normalised by the total number of regions within
                                     that bin. If this is combined with norm_value_by_region then each bin value
                                     is normalised by the total region coverage within that bin.
                        - "values_key": String. Specifies the value key to extract from the input feature file.
                                        This can be useful for something like DNA methylation where each basepair
                                        can be associated with multiple different values: number of unmethylated
                                        reads, number of methylated reads, percentage of reads methylated etc.
    - ID: Boolean. Indicates whether each region detailed in file is associated with some unique ID (for example,
          if the file were detailing gene expression from specific genes). If this is set to True then the
          function assumes that file also contains keys of the form:
                    "datatrack/{featurename}/ids/{chrom}"
          for each {chrom} in our chromosomes of interest and where {featurename} is some descriptive name of the
          feature. In this case the key value for these keys will be an (N_CHROM,) shape array containing IDs for
          each of the N_CHROM region in that chromosome.
    
    returns:
    
    - tvals: A binned datatrack array in which chromosomes 1,2,...,X are concatenated onto one another and each
             bin is given a value according to the total sum of the datatrack within each bin.
    - IDdf: If ID == True then this also returns a dataframe detailing the specific value of each region as well
            as the index of that region within the binned tvals array. 
    '''
    if ID:
        chrranges = {chrom: pd.DataFrame(np.arange((chrlims[chrom][1]- chrlims[chrom][0])/binSize)).T.astype(int) for chrom in chrlims}
        for chrom in chrranges:
            chrranges[chrom].columns = list(np.arange(chrlims[chrom][0] ,chrlims[chrom][1], binSize))
            
        IDdf, tvals = binDatatrackIDs(file,
                                      chrlims,
                                      binSize = binSize,
                                      input_params = input_params
                                     )
        return IDdf, tvals
    else:
        myparams = {'quantitative': True,
                    'values_key':'values',
                    'average': False,
                    'norm_value_by_region':True}
        fileparams = load_data_track(file, params = True)
        if fileparams is not None:
            try:
                for key in fileparams:
                    if key in myparams:
                        myparams[key] = fileparams[key]
            except:
                print("Couldn't read input file parameters. Make sure they are in dictionary form. Proceeding with default parameters")
        if input_params is not None:
            try:
                for key in input_params:
                    if key in myparams:
                        myparams[key] = input_params[key]
            except:
                print("Couldn't read input parameters. Make sure they are in dictionary form. Proceeding with default parameters")        
        r,v = load_data_track(file, values_key = myparams['values_key'])
        
        if not myparams['quantitative']:
            v = {chrom: np.ones(v[chrom].shape) for chrom in v}
        for chrom in r:
            for idx in np.arange(r[chrom].shape[0]):
                if abs(r[chrom][idx,1] - r[chrom][idx,0]) > 5e3:
                    mean = 0.5*((r[chrom][idx,1] + r[chrom][idx,0]))
                    r[chrom][idx, :] = np.array([mean - 1e3, mean + 1e3])
        t,r,v = binDataTrack(r,
                             v,
                             chrlims,
                             binSize = binSize,
                             norm_value_by_region = myparams['norm_value_by_region'])
        
        tvals = []                             
        chroms = [str(i+1) for i in np.arange(19)] + ['X']
        for chrom in chroms:
            tvals.append(t[chrom])
        
        tvals = np.concatenate(tvals)
        if myparams['average']:
            coverage = []
            c, _, _ = binDataTrack(r,
                                   {chrom: np.ones((v[chrom].shape[0],1)) for chrom in v},
                                   chrlims,
                                   binSize = binSize,
                                   norm_value_by_region = myparams['norm_value_by_region'])
            for chrom in chroms:
                coverage.append(c[chrom])
            
            coverage = np.concatenate(coverage)
            output = np.zeros(tvals.shape)
            output[coverage!= 0] = np.divide(tvals[coverage!=0], coverage[coverage!=0])
            tvals = output
    
        return tvals
