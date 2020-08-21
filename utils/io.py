from scipy.sparse import coo_matrix
from scipy import sparse
import torch
import pandas as pd
import math
import numpy as np
from numpy import int32

CHR_KEY_SEP = ' '

###############################################################################

def load_npz_contacts(file_path, 
        store_sparse=False,
        display_counts=False,
        normalize = False,
        cut_centromeres = True
        ):
    '''
    Utility function to load a .npz file containing contact information from a Hi-C experiment. 
    
    Arguments:
    
    - file_path: A .npz file generated using the nuc_tools ncc_bin tool. The function assumes a
                 File of this format
    - store_sparse: Boolean determining whether to return the contact matrices in sparse format
    - display_counts: Boolean determining whether to display summary plots of Hi-C counts
    - normalize: Boolean determining whether to normalise all matrix elements to lie between zero or one.
                 If False then raw contact counts are returned instead
    - cut_centromeres: Boolean determining whether to cut out the centromeres from the beginning of each
                       chromosome. Since the centromeres contain repetitive elements, they can't currently
                       be mapped by Hi-C so these rows and columns should be void of Hi-C contacts. This 
                       does affect indexing later on but other functions in this package should accommodate
                       for that
                       
    Returns:
    
    - bin_size: The size of each chromatin bin in basepairs.
    - chromo_limits: Dictionary detailing the start and end basepairs of each chromosome
                     in the contact dictionary. NOTE: the chromosome limits are inclusive
                     i.e. for each CHR_A we should have chromo_limits[CHR_A] = (start_A,
                     end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - contacts: Dictionary of matrices detailing contacts between chromosome pairs
    '''
    file_dict = np.load(file_path, allow_pickle=True, encoding = 'bytes')
  
    chromo_limits = {}
    contacts = {}
    bin_size, min_bins = file_dict['params']
    bin_size = int(bin_size*1e3)
  
    chromo_hists = {}
    cis_chromo_hists = {}

    pair_keys = [key for key in file_dict.keys() if "cdata" in key]
    nonpair_keys = [key for key in file_dict.keys() if (CHR_KEY_SEP not in key) and (key != 'params')]
  
    for key in nonpair_keys:
        offset, count = file_dict[key]
        chromo_limits[key] = offset*bin_size, (offset+count)*bin_size
        chromo_hists[key] = np.zeros(count)
        cis_chromo_hists[key] = np.zeros(count)

    maxc = 1
    if normalize:
        for key in sorted(pair_keys):
            maxc = np.maximum(maxc, np.max(file_dict[key]))

    for key in sorted(pair_keys):
        chr_a, chr_b, _ = key.split(CHR_KEY_SEP)
        shape = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "shape"]
        mtype = "CSR"
        try:
            indices = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "ind"]
            indptr = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "indptr"]
        except:
            mtype = "COO"
            row = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "row"]
            col = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "col"]

        if mtype == "CSR":

            mat = sparse.csr_matrix((file_dict[key]/maxc, indices, indptr), shape = shape)
        else:
            mat = sparse.coo_matrix((file_dict[key]/maxc, (row, col)), shape = shape)

        if not store_sparse:
            mat = mat.toarray()
          
        if chr_a == chr_b:
            a, b = mat.shape
            cols = np.arange(a-1)
            rows = cols-1

            if not np.all(mat[rows, cols] == mat[cols, rows]): # Not symmetric
                mat += mat.T
          
        contacts[(chr_a, chr_b)] = mat  
     
    #Chromosomes in our dataset
    chroms = chromo_limits.keys()
    if cut_centromeres:
    #Exclude centromeres of chromosomes where we don't have any contact data
        for chrom in chroms:
            chrmax = chromo_limits[chrom][-1]
            temp = contacts[(chrom, chrom)].indices
            chromo_limits[chrom] = (bin_size*np.min(temp[temp>0]), chrmax)
    
    for pair in contacts:
        s0, s1 = int(chromo_limits[pair[0]][0]/bin_size), int(chromo_limits[pair[1]][0]/bin_size)
        try:
            contacts[pair] = contacts[pair][s0:,s1:]
        except:
            contacts[pair] = contacts[pair].tocsr()[s0:, s1:].tocoo()
  
    if display_counts:
        # A simple 1D overview of count densities
 
        from matplotlib import pyplot as plt

        for chr_a, chr_b in contacts:
            mat = contacts[(chr_a, chr_b)]
            chromo_hists[chr_a] += mat.sum(axis=1)
            chromo_hists[chr_b] += mat.sum(axis=0)
 
            if chr_a == chr_b:
                cis_chromo_hists[chr_a] += mat.sum(axis=1)
                cis_chromo_hists[chr_b] += mat.sum(axis=0)
    
        all_sums = np.concatenate([chromo_hists[ch] for ch in chromo_hists])
        cis_sums = np.concatenate([cis_chromo_hists[ch] for ch in chromo_hists])
 
        fig, ax = plt.subplots()
 
        hist, edges = np.histogram(all_sums, bins=25, normed=False, range=(0, 500))
        ax.plot(edges[1:], hist, color='#0080FF', alpha=0.5, label='Whole genome (median=%d)' % np.median(all_sums))

        hist, edges = np.histogram(cis_sums, bins=25, normed=False, range=(0, 500))
        ax.plot(edges[1:], hist, color='#FF4000', alpha=0.5, label='Intra-chromo/contig (median=%d)' % np.median(cis_sums))
 
        ax.set_xlabel('Number of Hi-C RE fragment ends (%d kb region)' % (bin_size/1e3))
        ax.set_ylabel('Count')
 
        ax.legend()
 
        plt.show()

  
    return bin_size, chromo_limits, contacts


##################################################################################

def save_contacts(out_file_path, matrix_dict, chromo_limits, bin_size, min_bins=0):
    '''
    Save Hi-C a Hi-C contact dictionary to a .npz file. 
    
    Arguments:
    
    - out_file_path: Path to save the contact dictionary to
    - matrix_dict: Dictionary of Hi-C contacts. Dictionary should have keys of the form
                   (CHR_A, CHR_B). Trans contact matrices should be stored in sparse COO
                   format while cis contact matrices should be stored in sparse CSR
                   format.
    - chromo_limits: Dictionary detailing the start and end basepairs of each chromosome
                     in the contact dictionary. NOTE: the chromosome limits are inclusive
                     i.e. for each CHR_A we should have chromo_limits[CHR_A] = (start_A,
                     end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - bin_size: What is the size of each contact matrix bin in basepairs e.g. 50000
    - min_bins: Minimum number of bins to be included in a contig (read: chromosome) for it
                to be used downstream. 
    
    '''
    contacts = {}
    kb_bin_size = int(bin_size/1e3)
  
    for chr_a, chr_b in matrix_dict:
        pair = chr_a, chr_b
        key = CHR_KEY_SEP.join(pair)
  
        if chr_a == chr_b:
            contacts[key] = sparse.csr_matrix(matrix_dict[pair])
        else:
            contacts[key] = sparse.coo_matrix(matrix_dict[pair])
    
        start_a, end_a = chromo_limits[chr_a]
        start_b, end_b = chromo_limits[chr_b]
    
        min_a = int(start_a/bin_size)
        num_a = int(math.ceil(end_a/bin_size)) - min_a
        min_b = int(start_b/bin_size)
        num_b = int(math.ceil(end_b/bin_size)) - min_b
    
        # Store bin offsets and spans
        contacts[chr_a] = np.array([min_a, num_a])
        contacts[chr_b] = np.array([min_b, num_b])
    
        contacts['params'] = np.array([kb_bin_size, min_bins])    
  
    np.savez_compressed(out_file_path, **contacts) 
    

###################################################################
def load_data_track(file_path,
                    ID = False,
                    values_key = 'values',
                    params = False
                   ):
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
            print("Couldn't extract parameters from the file")
            return None
    
    data_archive = np.load(file_path, allow_pickle = True)
    regions_dict = {}
    values_dict = {}
    ID_dict = {}
    params = {}
    
    chromosomes = [str(i+1) for i in np.arange(19)] + ['X']
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
    
    if ID:
        return regions_dict, values_dict, ID_dict
    else:
        return regions_dict, values_dict
    
###################################################################
def save_data_track(dtrack, chrlims, binSize, out_path, track_name, ID = False):
    """Load track data (e.g. ChIp-seq)from Numpy archive (.npz)
    
    Arguments:
    
    - dtrack: Datatrack array. Assumes a 1-dimensional array of length N where N
              is the total number of bins across all chromosomes (effectively
              where chromosomes have been concatenated with one another starting
              with chromsome 1, 2 etc. and ending with chromosome X
    - chrlims: Dictionary detailing the start and end basepairs of each chromosome
               in the contact dictionary. NOTE: the chromosome limits are inclusive
               i.e. for each CHR_A we should have chromo_limits[CHR_A] = (start_A,
               end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A 
    - binSize: Size of each chromatin bin in basepairs
    - out_path: Path to save the datatrack to. 
    - track_name: Name of the datatrack e.g. Nanog_hapmESC for haploid mouse embryonic
                  stem cell Nanog ChIP-seq peaks
    
    """
   
    outdict = {}
    
    chromosomes = [str(i+1) for i in np.arange(19)] + ['X']
    if not ID:
        for chrom in chromosomes:
            regions = np.vstack([np.arange(chrlims[chrom][0], chrlims[chrom][1] + binSize, binSize),
                             np.arange(chrlims[chrom][0]+ binSize, chrlims[chrom][1] + 2*binSize, binSize)]).T
            values = dtrack[np.arange((chrlims[chrom][1] - chrlims[chrom][0])/binSize),:]
        
            key1 = "dtrack/regions/{}/{}".format(track_name, chrom)
            key2 = "dtrack/values/{}/{}".format(track_name, chrom)
            
            outdict[key1] = regions
            outdict[key2] = values
    else:
        try:
            for chrom in chromosomes:
                IDs = dtrack[chrom].columns.values
                regions = np.vstack([binSize*dtrack[chrom].values[0,:],
                                     (binSize+1)*dtrack[chrom].values[0,:]]).T
                values = binSize*dtrack[chrom].values[1,:]
                
                key1 = "dtrack/regions/{}/{}".format(track_name, chrom)
                key2 = "dtrack/values/{}/{}".format(track_name, chrom)
                key3 = "dtrack/id/{}/{}".format(track_name, chrom)
            
                outdict[key1] = regions
                outdict[key2] = values
                outdict[key3] = IDs
        except:
            print("If using ID == True then dtrack must be a dictionary containing pandas dataframes detailing the position and value of each ID")
            raise ValueError
                                                  
                
    np.savez(out_path, **outdict, allow_pickle = True)
    
    
#######################################################################
import utils.dtrack_utils as dtu 

def binDataTrack(regions,
                 values,
                 chr_lims,
                 void_path = '',
                 ID = None,
                 binSize = 1e5,
                 norm_value_by_region = True
                ):
    '''
    Bin a given datatrack (e.g. ChIP-seq) according to some chromosome limits and bin size. 
    
    Arguments:
    
    - regions: Dictionary detailing the regions of the genome with non-zero datatrack values.
               Each key of the dictionary should be a chromosome and each dictionary value 
               should be an (N,2) shape array where N is the number of regions on that
               chromosome and each row details the start and end of the region. Each region
               should be non-overlapping but the regions need not be ordered.
    - values: Dictionary detailing the datatrack values for each of the regions in the regions
              dictionary. That is, each key should be a chromosome and each key-value should be
              an (N,) shape array detailing the datatrack value for each region on that
              chromosome. NOTE: All genomic regions not contained in the regions dictionary
              are assume to have zero value.
    - chr_lims: Dictionary detailing the start and end basepairs of each chromosome
                in the contact dictionary. NOTE: the chromosome limits are inclusive
                i.e. for each CHR_A we should have chromo_limits[CHR_A] = (start_A,
                end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - void_path: Path to a 'void' datatrack detailing regions of the genome which should be
                 excluded from the binning procedure (i.e. given a datatrack value of zero).
                 If nothing is passed to binDataTrack there are assumed to be no void regions.
    - ID: If False then this is ignored. If not False this should be a dictionary containing the
          IDs of each datatrack region. 
    - binSize: The size of each chromatin bin in basepairs.
    - norm_value_by_region: Each region is associated with a value regardless of region width. 
                             If norm_values_by_region then the value associated with each region
                             is multiplied by {region_width}/{binSize}. Essentially this means
                             that a 10 basepair region with value 10 is treated the same as a 
                             100 basepair region with value 1 if both regions fall entirely within
                             the same bin. Think of this as enforcing an 'area under the curve'
                             style binning as opposed to weighting all regions equally.
    Returns:
    
    - binned_scores: A dictionary containing chromosomes as keys. Each key value is 
                     a (LEN_CHR,) shape array where LEN_CHR is the length of each
                     binned chromosome and the i-th array element is the datatrack
                     value for bin i. 
    '''
    chroms = [str(i+1) for i in np.arange(19)] + ['X']
    
    if void_path == '':
        void_regions = {chrom : np.array([]).astype(int32) for chrom in chroms}
    else:
        void_regions = load_data_track(void_path)[0]

    binned_scores = {key: [] for key in regions}

    out_regs = {}
    out_vals = {}
    if ID is not None:
        out_ID = {}
    for key in chroms:
        if key != 'Y':
            track_values = values[key][:,0]
            track_regions = regions[key]

            crc = np.array(track_regions).astype('int32')
            if len(void_regions[key]) > 0:
                idx = dtu.pairRegionsIntersection(crc,void_regions[key], exclude=True)
            else:
                idx = np.arange(len(crc))
                
            track_values = track_values[idx]
            out_vals[key] = track_values
            track_regions = crc[idx]
            out_regs[key] = track_regions
            if ID is not None:
                ID[key] = ID[key][idx]
                out_ID[key] = ID[key]
        
            chr_lim = chr_lims[key]

            if norm_value_by_region:
                nvbr = 1
            else:
                nvbr = 0
            binned_vals = dtu.regionBinValues(track_regions, 
                                              track_values, 
                                              binSize=binSize, 
                                              start = chr_lim[0], 
                                              end = chr_lim[-1],
                                              norm_value_by_region = int(nvbr)
                                             )
                    
            binned_scores[key] = binned_vals
    
    if not ID:
        return binned_scores, out_regs, out_vals
    else:
        return binned_scores, out_regs, out_vals, out_ID

############################################################################################################
def binDatatrackIDs(dataTrack_path, 
                    chr_lims,
                    void_path = '',
                    normby = False,
                    binSize = 1e5,
                    input_params = None
                   ):
    '''
    Bin a datatrack where each datatrack region has a unique ID and keep track of each ID. For example,
    when binning the transcriptional level from each chromatin bin, keep track of the transcript levels
    of each individual gene within each bin.
    
    Arguments:
    
    - dataTrack_path: Path to the datatrack .npz file
    - chr_lims: Dictionary detailing the start and end basepairs of each chromosome
                in the contact dictionary. NOTE: the chromosome limits are inclusive
                i.e. for each CHR_A we should have chromo_limits[CHR_A] = (start_A,
                end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - void_path: Path to a 'void' datatrack detailing regions of the genome which should be
                 excluded from the binning procedure (i.e. given a datatrack value of zero).
                 If nothing is passed to binDataTrack there are assumed to be no void regions.
    - normby: If not False then this should be a dataTrack to normalise our primary datatrack
              by.For example, if we are binning transcriptional values then we may want to 
              normalise each bin by the number of genes in that bin to get the average
              transcript value
    - binSize: The size of each chromatin bin in basepairs.
    - input_params: Dictionary with the keys specifying binning options. You need not specify all keys below but 
                    keys in the dictionary must come from:
                        - 'quantitative': Indicates whether the feature in question can be considered as
                                          quantitative or not. ChIPseq experiments without spike-in proteins
                                          produce peak heights which are hard to compare with one another and
                                          therefore we may not want to take into account peak height. In this
                                          case, all peak heights are treated as binary.
                        - 'norm_value_by_region': Each region is associated with a value regardless of region
                                                  width. If norm_values_by_region then the value associated
                                                  with each region is multiplied by {region_width}/{binSize}.
                                                  Essentially this means that a 10 basepair region with value
                                                  10 is treated the same as a 100 basepair region with value 1
                                                  if both regions fall entirely within the same bin. Think of
                                                  this as enforcing an 'area under the curve' style binning as
                                                  opposed to weighting all regions equally.
                        - 'average': If True, each bin value is normalised by the total number of regions within
                                     that bin. If this is combined with norm_value_by_region then each bin value
                                     is normalised by the total region coverage within that bin.
                        - 'values_key': String. Specifies the value key to extract from the input feature file.
                                        This can be useful for something like DNA methylation where each basepair
                                        can be associated with multiple different values: number of unmethylated
                                        reads, number of methylated reads, percentage of reads methylated etc.
    
    Returns:
    
    - outdf: Dataframe containing datatrack info for each unique ID. Columns
             of the dataframe are datatrack IDs. Row 1 contains the index of
             each datatrack ID in the vals array. Row 2 contains the
             datatrack value for each datatrack ID. 
    - vals: A dictionary containing chromosomes as keys. Each key value is 
            a (LEN_CHR,) shape array where LEN_CHR is the length of each
            binned chromosome and the i-th array element is the datatrack
            value for bin i. 
    '''
    chromosomes = [str(i+1) for i in np.arange(19)] + ['X']
    
    myparams = {'quantitative': True,
                    'values_key':'values',
                    'average': False,
                    'norm_value_by_region':True}
    fileparams = load_data_track(dataTrack_path, params = True)
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
                    
    regions, values, IDs = load_data_track(dataTrack_path, ID = True, values_key = myparams['values_key'])
        
    if not myparams['quantitative']:
        values = {chrom: np.ones(values[chrom].shape) for chrom in values}

    vals_dict, regions, values, IDs = binDataTrack(regions,
                                                   values,
                                                   chr_lims,
                                                   void_path,
                                                   ID = IDs,
                                                   binSize = binSize,
                                                   norm_value_by_region = myparams['norm_value_by_region']
                                                  )
    vals = [vals_dict[key] for key in chromosomes]
    vals = np.concatenate(vals)

    if normby:
        nregs, nvals = load_data_track(normby)
        nvals_dict, nregs, nvals = binDataTrack(nregs, nvals, chr_lims, void_path)
        nvals = [nvals_dict[key] for key in chromosomes]
        nvals = np.concatenate(nvals)
        vals = np.divide(vals, np.maximum(nvals-1, np.ones(np.shape(nvals))))

        
    IDdf0 = {key:[] for key in chromosomes}

    newIDdf = {}

    for key in chromosomes:
        mids = np.mean(regions[key], axis = 1)
        beadpos = int(binSize)*np.floor(mids/binSize).astype(int)
        myidxs = ((beadpos - chr_lims[key][0])/binSize).astype(int)
        
        useableidxs = (beadpos > chr_lims[key][0])&(beadpos < chr_lims[key][1])
        
        mids = mids[useableidxs]
        values[key] = values[key][useableidxs]
        IDs[key] = IDs[key][useableidxs]
        beadpos = beadpos[useableidxs]
        myidxs = myidxs[useableidxs]

        newIDdf[key] = pd.DataFrame(columns = IDs[key].T, data = np.append(myidxs[None,:],values[key][None,:], axis = 0))


    startingidxs = {str(i+1):0 for i in np.arange(19)}
    startingidxs['X'] = 0
    start = 0
    end = 0
    for key in chromosomes:
        startingidxs[key] = start
        start += int((chr_lims[key][1] - chr_lims[key][0])/binSize)

    for key in chromosomes:
        newIDdf[key].loc[0] += startingidxs[key]


    outdf = newIDdf['1']
    for key in chromosomes[1:]:
        outdf = outdf.join(newIDdf[key])

    return outdf, vals