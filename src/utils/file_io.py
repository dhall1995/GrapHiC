from scipy.sparse import coo_matrix
from scipy import sparse
import pandas as pd
import itertools
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
            if "chr" not in item:
                allowed_chroms[idx] = "chr" + item
    
    chrom_col = np.where(x.columns.values == chrom_col)[0][0]
    
    regions_dict = {}
    values_dict = {}
    ID_dict = {}
    
    per_chrom_regions = {k1: np.concatenate([item[None,[region_cols[0],region_cols[1]]] for item in list(g1)],axis = 0) for k1,g1 in itertools.groupby(sorted(x.values.astype('str'), key = lambda x:x[chrom_col]),lambda x: x[chrom_col])}
    
    if value_col is not None:
        per_chrom_values = {k1: np.concatenate([item[None,[value_col]] for item in list(g1)],axis = 0) for k1,g1 in itertools.groupby(sorted(x.values.astype('str'), key = lambda x:x[chrom_col]),lambda x: x[chrom_col])}
    else:
        per_chrom_values = {k1: value_fill*np.ones((per_chrom_regions[k1].shape[0],1)) for k1 in per_chrom_regions}
    
    if ID_col is not None:
        per_chrom_IDs = {k1: np.concatenate([item[None,[ID_col]] for item in list(g1)],axis = 0) for k1,g1 in itertools.groupby(sorted(x.values.astype('str'), key = lambda x:x[chrom_col]),lambda x: x[chrom_col])}
    else:
        per_chrom_IDs = None
    
    for k1 in per_chrom_regions:
        if "chr" not in str(k1):
            k1_ = "chr"+k1
        else:
            k1_ = k1
            
        if k1_ not in allowed_chroms:
            continue
        
        regions_dict[k1_] = per_chrom_regions[k1].astype('int32')
        values_dict[k1_] = per_chrom_values[k1]
        
        if per_chrom_IDs is not None:
            ID_dict[k1_] = per_chrom_IDs[k1]
            
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
        
    np.savez(out_path, **outdict)
    
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
    

######################################################################
import pickle
def save_obj(obj, out_path):
    with open(out_path +'.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(in_path):
    with open(in_path + '.pkl', 'rb') as f:
        return pickle.load(f)