import pandas as pd
import numpy as np
import glob
import cooler
import itertools
import os
from .utils.Datatrack import DataTrack_bigwig as dtbw

def evaluate_bigwigs_over_cooler_bins(cooler_path,
                                     bwpaths = [],
                                     names = [],
                                     stats_types = ['max'],
                                     allowed_chroms = ['chr{}'.format(idx+1) for idx in np.arange(19)]+['chrX'],
                                     verbose = True
                                    ):
    if len(names)!= len(bwpaths):
        names = bwpaths
        
    c = cooler.Cooler(cooler_path)
    bins = c.bins()
    chrom_binregs = {k1: np.concatenate([item[None,1:3] for item in list(g1)],
                                        axis = 0) for k1,g1 in itertools.groupby(sorted(bins[:].values,key = lambda x:x[0]),lambda x: x[0])}
    clr_index = np.concatenate([bins[:][['chrom']].values,
                               bins[:].index.values[:,None]],
                               axis = 1)
    chrom_stats = {k1: np.concatenate([item[None,[-1]] for item in list(g1)],
                                       axis = 0) for k1,g1 in itertools.groupby(sorted(clr_index,key = lambda x:x[0]),lambda x: x[0])}
    
    chrom_binregs = {chrom: chrom_binregs[chrom] for chrom in chrom_binregs if chrom in allowed_chroms}
    chrom_stats = {chrom: chrom_stats[chrom] for chrom in chrom_stats if chrom in allowed_chroms}
    colnames = []
    for idx, bigwig in enumerate(bwpaths):
        if verbose:
            print(names[idx], bigwig)
        bwname = os.path.split(bigwig)[1].split(".")[0]
        bw = dtbw('bigwig').from_bw(bigwig)
        for stype in stats_types:
            if verbose:
                print("\t{}".format(stype))
            for chrom in chrom_binregs:
                if chrom in bw.chr_lims:
                    stats_add = bw.bin_single_interval(chrom,
                                                   int(c.binsize),
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

def evaluate_dtrvp_over_cooler_bins(cooler_path,
                                    bedpaths = [],
                                    chrom_cols = [],
                                    region_cols = [],
                                    value_cols = [],
                                    stats_types = ['max'],
                                    allowed_chroms = ['chr{}'.format(idx+1) for idx in np.arange(19)]+['chrX'],
                                    verbose = True
                                    ):
    c = cooler.Cooler(cooler_path)
    bins = c.bins()
    chrom_binregs = {k1: np.concatenate([item[None,1:3] for item in list(g1)],
                                        axis = 0) for k1,g1 in itertools.groupby(sorted(bins[:].values,key = lambda x:x[0]),lambda x: x[0])}
    clr_index = np.concatenate([bins[:][['chrom']].values,
                               bins[:].index.values[:,None]],
                               axis = 1)
    chrom_stats = {k1: np.concatenate([item[None,[-1]] for item in list(g1)],
                                       axis = 0) for k1,g1 in itertools.groupby(sorted(clr_index,key = lambda x:x[0]),lambda x: x[0])}
    
    chrom_binregs = {chrom: chrom_binregs[chrom] for chrom in chrom_binregs if chrom in allowed_chroms}
    chrom_stats = {chrom: chrom_stats[chrom] for chrom in chrom_stats if chrom in allowed_chroms}
    names = []
    for idx, bed in enumerate(bedpaths):
        if verbose:
            print(bed)
        bedname = os.path.split(bed)[1].split(".")[0]
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
                                                   int(c.binsize),
                                                   interval = [chrom_binregs[chrom][0,0],
                                                               chrom_binregs[chrom][-1,1]],
                                                   stype = stype)[1][:,None]
                else:
                    stats_add = np.zeros(chrom_binregs[chrom].shape[0])[:,None]
                
                chrom_stats[chrom] = np.append(chrom_stats[chrom], 
                                           stats_add,
                                           axis = 1)
            names.append(bedname + "_{}".format(stype))
            
        
    chrom_stats = np.concatenate([chrom_stats[chrom] for chrom in allowed_chroms],
                                 axis = 0)
    
    return pd.DataFrame(data = chrom_stats[:,1:], 
                        index = chrom_stats[:,0], 
                        columns = names)

def evaluate_bigwigs_over_bed_dataframe(df,
                                        bwpaths = [],
                                        names = [],
                                        stats_types = ['max'],
                                        verbose = True):
    
    if len(names)!= len(bwpaths):
        names = bwpaths
        
    colnames = []
    chroms = df[df.columns.values[0]].unique()
    arr = []
    for idx, bigwig in enumerate(bwpaths):
        x = dtbw('x').from_bw(bigwig)
        for stype in stats_types:
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