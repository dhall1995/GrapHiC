from src.Graph_creation import (
    compute_ptg_graph_from_regions,
    add_binned_data_to_graphlist
)
from src.utils.misc import (
    buffer_regs, 
    ProgressBar
)

import cooler
import pandas as pd
import itertools
import numpy as np
import os
import argparse

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Construct feature matrices for all Hi-C bins within a cooler file, provided some bigwig files', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c","--cooler_file",
                        help="Cooler file from a Hi-C experiment",
                        default = '/disk2/dh486/cooler_files/WT/manually_downsampled/KR/KR_downsampled_WT_merged_10000.cool',
                        type=str)
    parser.add_argument("-r","--rnaseq",
                        help="path tsv with columns ['chrom','promoter_start','promoter_end','target','name']",
                        default = 'Data/rnaseq.tsv',
                        type=str)
    parser.add_argument("-b","--binned_data",
                        help="binned data created using make bigwig features",
                        default = 'Data/KR_downsampled_WT_merged_10000_bw_data.tsv',
                        type=str)
    parser.add_argument("-o","--out_path",
                        help="Folder to save graph objects to",
                        type = str,
                        default = 'Data/Graphs/')
    parser.add_argument("-ch","--chunk_size",
                        help="How many genes to try and process at once",
                        default = 500)
    args = parser.parse_args()
    
    
    c = cooler.Cooler(args.cooler_file)
    chr_lims = {str(row[0]):int(row[1]) for row in c.chroms()[:].values}
    
    rnaseq = pd.read_csv(args.rnaseq,sep = "\t")
    idxs = abs(rnaseq['lfc'].values) > np.percentile(abs(rnaseq['lfc'].values),75)
    rnaseq = rnaseq.loc[idxs]

    chr_lims = {str(row[0]):int(row[1]) for row in c.chroms()[:].values}

    prom_regions = {k1: np.concatenate([item[None,1:3] for item in list(g1)],
                                        axis = 0).astype('int32') for k1,g1 in itertools.groupby(sorted(rnaseq.values,key = lambda x:x[0]),lambda x: x[0])}

    target = {k1: np.concatenate([item[None,[3]] for item in list(g1)],
                                        axis = 0).astype('float')[:,0] for k1,g1 in itertools.groupby(sorted(rnaseq.values,key = lambda x:x[0]),lambda x: x[0])}
    names = {k1: np.concatenate([item[None,[4]] for item in list(g1)],
                                        axis = 0)[:,0] for k1,g1 in itertools.groupby(sorted(rnaseq.values,key = lambda x:x[0]),lambda x: x[0])}

    graph_regions = {chrom: buffer_regs(prom_regions[chrom],lims = [0,chr_lims[chrom]]) for chrom in prom_regions}
    
    ticker = 0
    total = rnaseq.shape[0]
    ProgressBar(ticker,
                total)
    for chrom in prom_regions:
        for idx in np.arange(0,
                             prom_regions[chrom].shape[0]-args.chunk_size,
                             args.chunk_size):
            rdict = {chrom: graph_regions[chrom][idx:idx+args.chunk_size,:]}
            ndict = {chrom: names[chrom][idx:idx+args.chunk_size]}
            tdict = {chrom: target[chrom][idx:idx+args.chunk_size]}
            prom_regs = {chrom: prom_regions[chrom][idx:idx+args.chunk_size,:]}
            glist = compute_ptg_graph_from_regions(args.cooler_file,
                                       rdict,
                                       names = ndict,
                                       balance = True,
                                       join = False,
                                       force_disjoint=False,
                                       record_cistrans_interactions = False,
                                       record_node_chromosome_as_onehot = False)
            add_binned_data_to_graphlist(glist[chrom],
                                         args.binned_data)
            for jdx,item in enumerate(glist[chrom]):
                item['target'] = tdict[chrom][jdx]
                item['prom_region'] = prom_regs[chrom][jdx,:]
                item['prom_chrom'] = chrom
                np.savez(os.path.join(args.out_path,"data_{}".format(ticker)), **item)
                ticker += 1
                ProgressBar(ticker,
                            total)
        
        rdict = {chrom: graph_regions[chrom][idx+args.chunk_size:,:]}
        ndict = {chrom: names[chrom][idx+args.chunk_size:]}
        tdict = {chrom: target[chrom][idx+args.chunk_size:]} 
        glist = compute_ptg_graph_from_regions(args.cooler_file,
                                       rdict,
                                       names = ndict,
                                       balance = True,
                                       join = False,
                                       force_disjoint=False,
                                       record_cistrans_interactions = False,
                                       record_node_chromosome_as_onehot = False)
        add_binned_data_to_graphlist(glist[chrom],
                                         args.binned_data)
        for jdx,item in enumerate(glist[chrom]):
            item['target'] = tdict[chrom][jdx]
            item['prom_region'] = prom_regs[chrom][jdx,:]
            item['prom_region'] = chrom
            np.savez(os.path.join(args.out_path,"data_{}".format(ticker)), **item)
            ticker += 1
            ProgressBar(ticker,
                        total)   
    
    

    