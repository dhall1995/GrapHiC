from scipy.sparse import coo_matrix
from scipy import sparse
import torch
import pandas as pd
import math
import numpy as np
from numpy import int32

CHROMS = [str(i+1) for i in np.arange(19)] + ['X']

###################################################################
def rvps_from_npz(file_path,
                  ID = False,
                  values_key = 'values',
                  params = False):
    """Load track data (e.g. ChIp-seq)from Numpy archive (.npz)
    
    Arguments:
    
    - file_path: Path of the data track to be loaded
    - ID: Boolean. Determines if each datatrack region contains a unique ID.
          For example, if the datatrack were genes or transcription then 
          each datatrack region is a specific gene with a specific ensemble
          ID.
    - values_key: Specifies which key to use within the .npz archive as our
                  datatrack values. If values_key doesn't exist in the
                  archive then the datatrack values are set to 1
    - params: Boolean. If true then search the data archive for a 'params'
              key and return that. The params dictionary is used to specify
              any default parameters to be used when binning a datatrack
          
    Returns:
    
    - regions_dict: Dictionary containing chromosomes as keys. Each key
                    value is an (N_CHR,2) shape array where each row is a
                    region and N_CHR is the number of non-zero datatrack
                    regions on that chromsome
    - values_dict: Dictionary containing chromosomes as keys. Each key 
                   value is an (N_CHR,) shape array detailing the
                   datatrack value for each non-zero datatrack region
    - ID_dict: If ID is True, returns a dictionary detailing the unique
               ID for each datatrack region.
    - params: If params is true, try to return the params dictionary from
              the data archive. If 'params' is not a key in archive then
              return an empty dictionary.
    """
    
    if params:
        data_archive = dict(np.load(file_path, allow_pickle = True))
        try:
            return data_archive['params'][()]
        except:
            print("Couldn't extract binning parameters from the file. ")
            return {}
    
    data_archive = np.load(file_path, allow_pickle = True)
    regions_dict = {}
    values_dict = {}
    ID_dict = {}
    params = {}
    
    for key in data_archive:
        if key != 'params':
            null, key2, track_name, chromo = key.split('/')
        
            if key2 == 'regions':
                regions_dict[chromo] = data_archive[key].astype('int32')
            elif key2 == values_key:
                try:
                    values_dict[chromo] = data_archive[key].astype('float')
                except:
                    reg_key = "/".join([null, 'regions', track_name, chromo])
                    num_regs = data_archive[reg_key].astype('int32').shape[0]
                    values_dict[chromo] = np.zeros((num_regs,1)).astype('float')        
            elif ID and key2 == 'id':
                ID_dict[chromo] = data_archive[key]
    

    return regions_dict, values_dict, ID_dict

###################################################################
def rvps_from_bed(file_path,
                  chrom_col = 0,
                  region_cols = (1,2),
                  value_col = None,
                  ID_col = None,
                  value_fill = 1,
                  header = None,
                  allowed_chroms = None,
                  sep = "\t"):
    """Load track data (e.g. ChIp-seq)from Numpy archive (.npz)
    
    Arguments:
    
    - file_path: Path of the data track to be loaded (bed format)
    - chrom_col: int. Column of the bed file containing the chromosome information
    - region_cols: 2-tuple. Columns of the bed file containing the regions for
                   each value.
    - value_col: int. Column of the bed file containing the value for each region.
                 If this is None then each region is given a score given by the
                 value_fill input argument.
    - ID_col: int. If each region has a specific ID associated with it then
              this is stored in an ID dictionary along with the regions
    - value_fill: float. If value_col is None then we give each region a value
                  according to the value_fill input.
    - header: None. If the bed file has a header then we ignore line 0 and 
              skip to line 1.
    - allowed_chroms: List of chromosomes which we want to include in our datatrack dictionaries.
                        if None then all chromosomes are allowed. 
    - sep: Separating values in the bed file.
                  
          
    Returns:
    
    - regions_dict: Dictionary containing chromosomes as keys. Each key
                    value is an (N_CHR,2) shape array where each row is a
                    region and N_CHR is the number of non-zero datatrack
                    regions on that chromsome
    - values_dict: Dictionary containing chromosomes as keys. Each key 
                   value is an (N_CHR,) shape array detailing the
                   datatrack value for each non-zero datatrack region
    - ID_dict: If ID is True, returns a dictionary detailing the unique
               ID for each datatrack region.
    """
    
    
    x = pd.read_csv(file_path, sep = sep, header = header)
    
    if allowed_chroms is None:
        allowed_chroms = list(set(x[chrom_col].values))
        for idx, item in enumerate(allowed_chroms):
            #use the chromosome naming convention that chromosomes don't start with chr
            if "chr" in item:
                allowed_chroms[idx] = item[3:]
        
    regions_dict = {}
    values_dict = {}
    ID_dict = {}
    for idx in np.arange(x.values.shape[0]):
        chrom = x.loc[idx][0]
        if "chr" in chrom:
            chrom = chrom[3:]
        if chrom not in allowed_chroms:
            continue
            
        start = x.loc[idx][region_cols[0]]
        end = x.loc[idx][region_cols[1]]
        if value_col is not None:
            val = x.loc[idx][value_col]
        else:
            val = value_fill
            
        if ID_col is not None:
            ID = x.loc[idx][ID_col]
    
        if chrom not in regions_dict:
            regions_dict[chrom] = [[start, end]]
            values_dict[chrom] = [[val]]
            if ID_col is not None:
                ID_dict[chrom] = [[ID]]
        else:
            regions_dict[chrom].append([start, end])
            values_dict[chrom].append([val])
            if ID_col is not None:
                ID_dict[chrom].append([ID])
        
    for key in regions_dict:
        regions_dict[key] = np.array(regions_dict[key])
        values_dict[key] = np.array(values_dict[key])
        if ID_col is not None:
            ID_dict[key] = np.array(ID_dict[key])
            
    return regions_dict, values_dict, ID_dict
    
###################################################################
def rvps_to_npz(regions,
                values,
                track_name,
                out_path,
                IDs = None,
                params = None
               ):
    """Save track data (e.g. ChIp-seq) to  Numpy archive (.npz)
    Arguments:
    - regions: dictionary detialing an (N_chrom,2) shape array detailing the
               regions of the datatrack for each chromosome.
    - values: dictionary detialing an (N_chrom,) shape array detailing the values
              associated with each region.
    - track_name: Descriptive name of the datatrack.
    - out_path: The path to save the .npz archive to.
    - IDs: If each regions is associated with a unique ID then save these IDs
           in a dictionary with an (N_chrom,) shape array for each chromosome.
    """
   
    outdict = {}
    
    for chrom in regions:    
        key1 = "dtrack/regions/{}/{}".format(track_name, chrom)
        key2 = "dtrack/values/{}/{}".format(track_name, chrom)
        if IDs is not None:
            key3 = "dtrack/id/{}/{}".format(track_name, chrom)
            
        outdict[key1] = regions[chrom]
        if chrom in values:
            outdict[key2] = values[chrom]
        else:
            print("Couldn't find chromosome {} in values. Assuming ones instead".format(chrom))
            outdict[key2] = np.ones(regions[chrom].shape(0))
        if IDs is not None:
            outdict[key3] = IDs
    
    if params is not None:
        outdict['params'] = params
        
    np.savez(out_path, **outdict, allow_pickle = True)
    
###################################################################
def rvps_to_bed(regions,
                values,
                track_name,
                out_path,
                IDs = None,
                sep = "\t"
               ):
    """Save track data (e.g. ChIp-seq) to  Numpy archive (.npz)
    Arguments:
    - regions: dictionary detialing an (N_chrom,2) shape array detailing the
               regions of the datatrack for each chromosome.
    - values: dictionary detialing an (N_chrom,) shape array detailing the values
              associated with each region.
    - track_name: Descriptive name of the datatrack.
    - out_path: The path to save the .npz archive to.
    - IDs: If each regions is associated with a unique ID then save these IDs
           in a dictionary with an (N_chrom,) shape array for each chromosome.
    """
    if IDs is None:
        IDs = {}
        
    ID_counter = 0
    with open(out_path,'w') as op:
        for chromosome in regions:
            for idx in np.arange(regions[chromosome].shape[0]):
                region = regions[chromosome][idx,:]
                value = values[chromosome][idx]
                try:
                    ID = IDs[chromosome][idx]
                except:
                    ID = "chr{}_{}_{}".format(chromosome,track_name, ID_counter)
                    ID_counter += 1
                
                op.write(sep.join([chromosome, region[0], region[1],value, ID]))
                op.write("\n")
                
        
###################################################################
def bed_to_npz(bed_path,
               out_path,
               track_name,
               chrom_col = 0,
               region_cols = (1,2),
               value_col = None,
               ID_col = None,
               value_fill = 1,
               header = None,
               allowed_chroms = None,
               sep = "\t",
               params = None):
    '''
    convert a BED format region-value-pairs input to a .npz archive
    '''
    regions_dict, values_dict, ID_dict = rvps_from_bed(file_path,
                                                      chrom_col = chrom_col,
                                                      region_cols = region_cols,
                                                      value_col = value_col,
                                                      ID_col = ID_col,
                                                      value_fill = value_fill,
                                                      header = header,
                                                      allowed_chroms = allowed_chroms,
                                                      sep = sep)
    if len(list(ID_dict.keys())) == 0:
        ID_dict = None
    
    rvps_to_npz(regions_dict,
                values_dict,
                track_name,
                out_path,
                IDs = ID_dict,
                params = params)
    


##############################################################################################

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



######################################################################
import pickle
def save_obj(obj, out_path):
    with open(out_path +'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(in_path):
    with open(in_path + '.pkl', 'rb') as f:
        return pickle.load(f)