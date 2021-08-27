import pandas as pd
import numpy as np
import glob
import cooler
import itertools
import os
from .utils.Datatrack import DataTrack_bigwig as dtbw
from .utils.Datatrack import DataTrack_rvp as dtrvp

from typing import List, Dict, Tuple

CHROMS = ['chr{}'.format(idx+1) for idx in np.arange(19)]+['chrX']

def cooler_bin_info(cooler_path: str,
                    allowed_chroms:List = CHROMS
                   )->Tuple:
    """
	Retrieve bin-level information from a cooler object

	:param cooler_path: path to cooler file
	:type cooler_path: str
	:param allowed_chroms: List of chromosomes to retrieve from the cooler file
	:type allowed_chroms: List
	:return: Tuple containing a Dictionary, chrom_binregs, of regions associated with each bin; a Dictionary, chrom_stats, of cooler indices associated with each bin; an binsize
	:rtype: tuple
        """
    c = cooler.Cooler(cooler_path)
    bins = c.bins()
    binsize = int(c.binsize)
    chrom_binregs = {k1: np.concatenate([item[None,1:3] for item in list(g1)],
                                        axis = 0) for k1,g1 in itertools.groupby(sorted(bins[:].values,key = lambda x:x[0]),lambda x: x[0])}
    clr_index = np.concatenate([bins[:][['chrom']].values,
                               bins[:].index.values[:,None]],
                               axis = 1)
    chrom_stats = {k1: np.concatenate([item[None,[-1]] for item in list(g1)],
                                       axis = 0) for k1,g1 in itertools.groupby(sorted(clr_index,key = lambda x:x[0]),lambda x: x[0])}
    
    chrom_binregs = {chrom: chrom_binregs[chrom] for chrom in chrom_binregs if chrom in allowed_chroms}
    chrom_stats = {chrom: chrom_stats[chrom] for chrom in chrom_stats if chrom in allowed_chroms}
    
    return chrom_binregs, chrom_stats, binsize
    
def evaluate_tracks_over_cooler_bins(cooler_path:str,
                                     paths: List = [],
                                     names: List = [],
                                     stats_types: List[str] = ['max'],
                                     allowed_chroms: List = CHROMS,
                                     value_col: int = 3,
                                     region_cols: Tuple[int] = (1,2),
                                     chrom_col: int = 0,
                                     verbose: bool = True
                                    )->pd.DataFrame:
    """
    Evaluate multiple tracks over all bins within a cooler object and return the results in a Pandas dataframe
    
    :param cooler_path: path to cooler file
    :type cooler_path: str
    :param paths: List of paths to (multiple) bigwig or BED files 
    :type paths: List
    :param names: List of names to associated with each datatrack provided with path. If the length of the names list doesn't equal the length of the paths list then the function instead assigns names based on filenames 
    :type names: List
    :param stats_types: List of statistics to collect over each bin. Allowed statistics are: mean, max, min, sum, coverage, std 
    :type stats_types: List[str]
    :param allowed_chroms: List of chromosomes to retrieve from the cooler file
    :type allowed_chroms: List
    :param value_col: Which collumn to collect values from in provided BED files (default = 3) (Note: zero-indexing is assumed)
    :type value_col: int
    :param region_cols: Tuple detailing which columns (zero-indexed) to collect the region information from in provided BED files (default = (1,2))
    :type region_cols: tuple
    :param chrom_col: Which collumn to collect chromosome information from in provided BED files (default = 0) (Note: zero-indexing is assumed)
    :type chrom_col: int
    :param verbose: Whether to print progress/names etc. 
    :type verbose: bool
    :return: Dataframe detailing evaluated tracks/statistics over cooler bins
    :rtype: pd.DataFrame
    """
    if len(names)!= len(paths):
        names = [os.path.split(path)[1].split(".")[0] for path in paths]
    
    chrom_binregs, chrom_stats, binsize = cooler_bin_info(cooler_path, 
                                                          allowed_chroms = allowed_chroms)
        
    colnames = []
    for idx, path in enumerate(paths):
        if verbose:
            print(names[idx], path)
        try:
            track = dtbw('bigwig').from_bw(path)
        except:
            track = dtrvp('dtrvp').from_bed(path,
                                            chrom_col=chrom_col,
                                            region_cols = region_cols,
                                            value_col = value_col)
            track.chrlims_from_regions()
        for stype in stats_types:
            if verbose:
                print("\t{}".format(stype))
            for chrom in chrom_binregs:
                if chrom in track.chr_lims:
                    stats_add = track.bin_single_interval(chrom,
                                                          binsize,
                                                          interval = [chrom_binregs[chrom][0,0],
                                                                      chrom_binregs[chrom][-1,1]],
                                                          stats_type = stype,
                                                          exact = True)[1][:,None]
                else:
                    stats_add = np.zeros(chrom_binregs[chrom].shape[0])[:,None]
                
                chrom_stats[chrom] = np.append(chrom_stats[chrom], 
                                           stats_add,
                                           axis = 1)
            colnames.append(names[idx] + "_{}".format(stype))
            
        
    chrom_stats = np.concatenate([chrom_stats[chrom] for chrom in allowed_chroms],
                                 axis = 0)
    
    return pd.DataFrame(data = chrom_stats[:,1:], 
                        index = chrom_stats[:,0], 
                        columns = colnames)

def evaluate_bigwigs_over_cooler_bins(cooler_path: str,
                                     bwpaths: List[str] = [],
                                     names: List[str] = [],
                                     stats_types: List[str] = ['max'],
                                     allowed_chroms: List = CHROMS,
                                     verbose: bool = True
                                    )->pd.DataFrame:
    """
    Evaluate multiple bigwigs over all bins within a cooler object and return the results in a Pandas dataframe
    
    :param cooler_path: path to cooler file
    :type cooler_path: str
    :param bwpaths: List of paths to (multiple) bigwig files 
    :type bwpaths: List[str]
    :param names: List of names to associated with each datatrack provided with path. If the length of the names list doesn't equal the length of the paths list then the function instead assigns names based on filenames 
    :type names: List[str]
    :param stats_types: List of statistics to collect over each bin. Allowed statistics are: mean, max, min, sum, coverage, std 
    :type stats_types: List[str]
    :param allowed_chroms: List of chromosomes to retrieve from the cooler file
    :type allowed_chroms: List
    :param verbose: Whether to print progress/names etc. 
    :type verbose: bool
    :return: Dataframe detailing evaluated tracks/statistics over cooler bins
    :rtype: pd.DataFrame
    """
    if len(names)!= len(bwpaths):
        names = [os.path.split(bigwig)[1].split(".")[0] for bigwig in bwpaths]
        
    chrom_binregs, chrom_stats, binsize = cooler_bin_info(cooler_path, 
                                                          allowed_chroms = allowed_chroms)
        
    colnames = []
    for idx, bigwig in enumerate(bwpaths):
        if verbose:
            print(names[idx], bigwig)
        bw = dtbw('bigwig').from_bw(bigwig)
        for stype in stats_types:
            if verbose:
                print("\t{}".format(stype))
            for chrom in chrom_binregs:
                if chrom in bw.chr_lims:
                    stats_add = bw.bin_single_interval(chrom,
                                                   binsize,
                                                   interval = [chrom_binregs[chrom][0,0],
                                                               chrom_binregs[chrom][-1,1]],
                                                   stats_type = stype,
                                                       exact = True)[1][:,None]
                else:
                    stats_add = np.zeros(chrom_binregs[chrom].shape[0])[:,None]
                
                chrom_stats[chrom] = np.append(chrom_stats[chrom], 
                                           stats_add,
                                           axis = 1)
            colnames.append(names[idx] + "_{}".format(stype))
            
        
    chrom_stats = np.concatenate([chrom_stats[chrom] for chrom in allowed_chroms],
                                 axis = 0)
    
    return pd.DataFrame(data = chrom_stats[:,1:], 
                        index = chrom_stats[:,0], 
                        columns = colnames)

def evaluate_dtrvp_over_cooler_bins(cooler_path: str,
                                    bedpaths: List[str] = [],
                                    names: List[str] = [],
                                    stats_types: List[str] = ['max'],
                                    value_cols: List = [],
                                    region_cols: List = [],
                                    chrom_cols: List = [],
                                    allowed_chroms: List = CHROMS,
                                    verbose: bool = True
                                    )-> pd.DataFrame:
    """
    Evaluate multiple BED (dtrvp = datatrack region-value-pairs) style tracks over all bins within a cooler object and return the results in a Pandas dataframe
    
    :param cooler_path: path to cooler file
    :type cooler_path: str
    :param bedpaths: List of paths to (multiple) BED files 
    :type bedpaths: List
    :param names: List of names to associated with each datatrack provided with path. If the length of the names list doesn't equal the length of the paths list then the function instead assigns names based on filenames 
    :type names: List
    :param stats_types: List of statistics to collect over each bin. Allowed statistics are: mean, max, min, sum, coverage, std 
    :type stats_types: List[str]
    :param value_cols: List detailing which collumns to collect values from in provided BED files (Note: zero-indexing is assumed and different value columns can be specified per BED file unlike evaluate_tracks_over_cooler_bins)
    :type value_cols: int
    :param region_cols: List of tuples detailing which columns (zero-indexed) to collect the region information from in provided BED files. Different region columns can be specified per BED file unlike evaluate_tracks_over_cooler_bins.
    :type region_cols: tuple
    :param chrom_cols: List detailing which collumn to collect chromosome information from in provided BED files (Note: zero-indexing is assumed and different chromosome columns can be specified per BED file unlike evaluate_tracks_over_cooler_bins)
    :type chrom_cols: int
    :param allowed_chroms: List of chromosomes to retrieve from the cooler file
    :type allowed_chroms: List
    :param verbose: Whether to print progress/names etc. 
    :type verbose: bool
    :return: Dataframe detailing evaluated tracks/statistics over cooler bins
    :rtype: pd.DataFrame
    """
    if len(names)!= len(bedpaths):
        names = [os.path.split(bed)[1].split(".")[0] for bed in bedpaths]
        
    chrom_binregs, chrom_stats, binsize = cooler_bin_info(cooler_path, 
                                                          allowed_chroms = allowed_chroms)
        
    names = []
    for idx, bed in enumerate(bedpaths):
        if verbose:
            print(bed)
        rvp = dtrvp('dtrvp').from_bed(bed, 
                                      chrom_col=chrom_cols[idx], 
                                      region_cols = region_cols[idx,:], 
                                      value_col = value_cols[idx])
        for stype in stats_types:
            if verbose:
                print("\t{}".format(stype))
            for chrom in chrom_binregs:
                if chrom in rvp.chr_lims:
                    stats_add = rvp.bin_single_interval(chrom,
                                                   binsize,
                                                   interval = [chrom_binregs[chrom][0,0],
                                                               chrom_binregs[chrom][-1,1]],
                                                   stype = stype)[1][:,None]
                else:
                    stats_add = np.zeros(chrom_binregs[chrom].shape[0])[:,None]
                
                chrom_stats[chrom] = np.append(chrom_stats[chrom], 
                                           stats_add,
                                           axis = 1)
            names.append(names[idx] + "_{}".format(stype))
            
        
    chrom_stats = np.concatenate([chrom_stats[chrom] for chrom in allowed_chroms],
                                 axis = 0)
    
    return pd.DataFrame(data = chrom_stats[:,1:], 
                        index = chrom_stats[:,0], 
                        columns = names)

def evaluate_tracks_over_bed_dataframe(df: pd.DataFrame,
                                       paths: List[str] = [],
                                       names: List[str] = [],
                                       stats_types: List[str] = ['max'],
                                       value_col: int = 3,
                                       region_cols: tuple = (1,2),
                                       chrom_col: int = 0,
                                       verbose: bool = True):
    """
    Evaluate multiple BED or bigwig style tracks over an arbitrary BED style dataframe in which the 0th column details the chromosome and the 1nd and 2nd column detail the regions. Contrary to evaluate_tracks_over_cooler_bins, this instead returns both new column names and a value array which can then be appended to the original BED-style dataframe for for ease of access later.
    
    :param df: BED style DataFrame
    :type df: pd.DataFrame
    :param paths: List of paths to (multiple) bigwig or BED files 
    :type paths: List
    :param names: List of names to associated with each datatrack provided with path. If the length of the names list doesn't equal the length of the paths list then the function instead assigns names based on filenames 
    :type names: List
    :param stats_types: List of statistics to collect over each bin. Allowed statistics are: mean, max, min, sum, coverage, std 
    :type stats_types: List[str]
    :param value_col: Which collumn to collect values from in provided BED files (default = 3) (Note: zero-indexing is assumed)
    :type value_col: int
    :param region_cols: Tuple detailing which columns (zero-indexed) to collect the region information from in provided BED files (default = (1,2))
    :type region_cols: tuple
    :param chrom_col: Which collumn to collect chromosome information from in provided BED files (default = 0) (Note: zero-indexing is assumed)
    :type chrom_col: int
    :param verbose: Whether to print progress/names etc. 
    :type verbose: bool
    :return: list of column names of length len(paths) and a value array of shape (df.shape[0],len(paths))
    :rtype: list, array
    """
    
    if len(names)!= len(paths):
        names = [os.path.split(path)[1].split(".")[0] for path in paths]
        
    colnames = []
    chroms = df[df.columns.values[0]].unique()
    dfchromcol = df.columns.values[0]
    arr = []
    for idx, path in enumerate(paths):
        if verbose:
            print(names[idx], path)
        try:
            track = dtbw('bigwig').from_bw(path)
        except:
            track = dtrvp('dtrvp').from_bed(path,
                                            chrom_col=chrom_col,
                                            region_cols = region_cols,
                                            value_col = value_col)
            track.chrlims_from_regions()
        for stype in stats_types:
            if verbose:
                print("\t{}".format(stype))
            vals = np.zeros(df.shape[0])
            for chrom in chroms:
                idxs = df[dfchromcol]==chrom
                regs = df.loc[idxs,
                              [df.columns.values[1],
                               df.columns.values[2]
                              ]
                             ].values
                stats = track.stats(chrom,
                                    regs.astype('int32'),
                                    stats_type=stype)
                vals[idxs.values] = stats
            arr.append(vals.astype('float'))
            colnames.append(names[idx] + "_{}".format(stype))
    
    arr = np.concatenate([item[:,None] for item in arr],
                         axis = 1)
    
    return colnames, arr


def evaluate_bigwigs_over_bed_dataframe(df: pd.DataFrame,
                                        bwpaths: List = [],
                                        names: List = [],
                                        stats_types: List[str] = ['max'],
                                        verbose: bool = True):
    """
    Evaluate multiple bigwig style tracks over an arbitrary BED style dataframe in which the 0th column details the chromosome and the 1nd and 2nd column detail the regions. Contrary to evaluate_tracks_over_cooler_bins, this instead returns both new column names and a value array which can then be appended to the original BED-style dataframe for for ease of access later.
    
    :param df: BED style DataFrame
    :type df: pd.DataFrame
    :param bwpaths: List of paths to (multiple) bigwig files 
    :type bwpaths: List
    :param names: List of names to associated with each datatrack provided with path. If the length of the names list doesn't equal the length of the paths list then the function instead assigns names based on filenames 
    :type names: List
    :param stats_types: List of statistics to collect over each bin. Allowed statistics are: mean, max, min, sum, coverage, std 
    :type stats_types: List[str]
    :param verbose: Whether to print progress/names etc. 
    :type verbose: bool
    :return: list of column names of length len(paths) and a value array of shape (df.shape[0],len(paths))
    :rtype: list, array
    """
    
    if len(names)!= len(bwpaths):
        names = bwpaths
        
    colnames = []
    chroms = df[df.columns.values[0]].unique()
    arr = []
    for idx, bigwig in enumerate(bwpaths):
        if verbose:
            print(names[idx], bigwig)
        x = dtbw('x').from_bw(bigwig)
        for stype in stats_types:
            if verbose:
                print("\t{}".format(stype))
            vals = np.zeros(df.shape[0])
            for chrom in chroms:
                idxs = df['chromosome']==chrom
                regs = df.loc[idxs,
                              [df.columns.values[1],
                               df.columns.values[2]
                              ]
                             ].values
        
                vals[idxs.values] = x.stats(chrom,
                                        regs.astype('int32'),
                                        stats_type=stype)[:,0]
            arr.append(vals.astype('float'))
            colnames.append(names[idx] + "_{}".format(stype))
    
    arr = np.concatenate([item[:,None] for item in arr],
                         axis = 1)
    
    return colnames, arr
