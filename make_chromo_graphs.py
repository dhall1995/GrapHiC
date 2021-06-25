from src.Graph_creation import (
    from_regions,
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
from functools import partial
from multiprocessing import Pool

CHROMS = [f'chr{idx+1}' for idx in np.arange(19)]+['chrX']

def mytransform(vec,perc=95):
    absvec = abs(vec)
    vperc = np.percentile(absvec[absvec>0],
                          perc)
    out = np.divide(vec,vperc)
    out[out>1] = 1
    
    return out

def name_chr(chrom):
    if 'chr' in chrom:
        return chrom
    else:
        return "chr"+chrom
    
def make_graph(clrs, 
               binned_data, 
               regions, 
               chrom):
    gdict = from_regions(clrs,
                         {chrom: regions[chrom]},
                         balance = True,
                         join = False,
                         force_disjoint=False,
                         record_cistrans_interactions = False,
                         record_node_chromosome_as_onehot = False)
    
    add_binned_data_to_graphlist(gdict[chrom],
                                 binned_data)
        
    return gdict[chrom][0],chrom
    
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Construct feature matrices for all Hi-C bins within a cooler file, provided some bigwig files', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c","--cooler_files",
                        nargs = "+",
                        help="Cooler file from a Hi-C experiment",
                        default = '/disk2/dh486/cooler_files/WT/manually_downsampled/KR/KR_downsampled_WT_merged_10000.cool',
                        type=str)
    parser.add_argument("-ch","--chromosomes",
                        nargs = "+",
                        help="Chromsomes to retrieve",
                        default = CHROMS,
                        type = str)
    parser.add_argument("-tr","--tracks",
                        help="track paths",
                        nargs = "+",
                        default = ['Mbd3_ESC.bw'],
                        type=str)
    parser.add_argument("-tn","--track_names",
                        help="names of each datatrack",
                        nargs = "+",
                        default = None,
                        type=str)
    parser.add_argument("-s","--statistics_types",
                        help="Statistics of each track to collect over each specified region and cooler bin",
                        nargs = "+",
                        default = 'max',
                        type=str)
    parser.add_argument("-tf","--transform",
                        help="Transform to apply to the data. Can either provide a custom transform as in this script or specify 'robust', 'standard' or 'power' to use a scikit-learn defined data transform. Not specifying will result in no transform.",
                        default = None)
    parser.add_argument("-o","--out_path",
                        help="Folder to save graph objects to",
                        type = str,
                        default = 'Data/Chromosomes/')
    args = parser.parse_args()
    
    if isinstance(args.cooler_files,str):
        args.cooler_files = [args.cooler_files]
    
    if isinstance(args.chromosomes,str):
        args.chromosomes = [args.chromosomes]

    #EVALUATE BIGWIGS OVER COOLER BINS AND SAVE TO FILE
    track_out_file = os.path.join(self.processed_dir, "cooler_track_data.csv")
    print(f"Evaluating tracks over cooler bins and saving to file {track_out_file}")
    df = eval_tracks_over_cooler(args.cooler_files[0],
                                  paths = args.tracks,
                                  names = args.track_names,
                                  stats_types = args.statistic_types,
                                  allowed_chroms = self.chromosomes
                                 )
        
    if isinstance(args.transform,str):
        if args.transform == "Power":
            norm = PowerTransform_norm
        elif args.transform == "Standard":
            norm = Standard_norm
        elif args.transform == "Robust":
            norm = Robust_norm
    elif args.transform is not None:
        norm = lambda x: np.apply_along_axis(mytransform,
                                                 0,
                                                 x)
            
    if self.track_transform is not None:
            df = pd.DataFrame(data= norm(df.values.astype('float')),
                              columns = df.columns,
                              index= df.index)
        
        df.to_csv(track_out_file, 
                  sep = "\t")
    
    c = cooler.Cooler(args.cooler_files[0])
    chr_lims = {str(row[0]):int(row[1]) for row in c.chroms()[:].values}

    graph_regions = {name_chr(chrom): np.array([[0,chr_lims[name_chr(chrom)]]]) for chrom in args.chromosomes}
    
    ticker = 0
    ProgressBar(ticker,
                len(args.chromosomes)
               )
                
    fn = partial(make_graph, args.cooler_files, df, graph_regions)
    p = Pool()
    t_outs = p.imap(fn, (chrom for chrom in graph_regions))
    for t_out in t_outs:
        graph, chrom = t_out
        np.savez(os.path.join(args.out_path,"chromosome_{}".format(chrom)), **graph)
        ticker += 1
        ProgressBar(ticker,
                    len(args.chromosomes))

    