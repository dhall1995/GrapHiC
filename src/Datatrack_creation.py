import pandas as pd
import numpy as np
import glob
import cooler
import itertools
import os
from .utils.Datatrack import DataTrack_bigwig as dtbw

def evaluate_bigwig_over_cooler_bins(cooler_path,
                                     bwpaths = [],
                                     stats_types = ['max'],
                                     allowed_chroms = ['chr{}'.format(idx+1) for idx in np.arange(19)]+['chrX']
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
    for bigwig in bwpaths:
        bwname = os.path.split(bigwig)[1].split(".")[0]
        bw = dtbw('bigwig').from_bw(bigwig)
        for stype in stats_types:
            for chrom in chrom_binregs:
                if chrom in bw.chr_lims:
                    stats_add = bw.bin_single_interval(chrom,
                                                   int(c.binsize),
                                                   interval = [chrom_binregs[chrom][0,0],
                                                               chrom_binregs[chrom][-1,1]],
                                                   type = stype,
                                                       exact = True)[1][:,None]
                else:
                    stats_add = np.zeros(chrom_binregs[chrom].shape[0])[:,None]
                
                chrom_stats[chrom] = np.append(chrom_stats[chrom], 
                                           stats_add,
                                           axis = 1)
            names.append(bwname + "_{}".format(stype))
            
        
    chrom_stats = np.concatenate([chrom_stats[chrom] for chrom in allowed_chroms],
                                 axis = 0)
    
    return pd.DataFrame(data = chrom_stats[:,1:], 
                        index = chrom_stats[:,0], 
                        columns = names)