from GrapHiC.Dataset import HiC_Dataset
import numpy as np

CHROMS = [f'chr{idx+1}' for idx in np.arange(19)]+['chrX']

def mytransform(vec,perc=95):
    absvec = abs(vec)
    vperc = np.percentile(absvec[absvec>0],
                          perc)
    out = np.divide(vec,vperc)
    out[out>1] = 1
    
    return out

if __name__=="__main__":
    
    import argparse
    parser = argparse.ArgumentParser(description='Construct a datset of per-gene graph objects based off contact info, some target (e.g. rnaseq) and protein binding data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c","--cooler_files",
                        nargs = "+",
                        help="Cooler file(s) from a Hi-C experiment(s)",
                        default = 'KR_downsampled_WT_merged_10000.cool',
                        type=str)
    parser.add_argument("-t","--target",
                        help="path tsv with columns ['chrom','promoter_start','promoter_end','target','name']",
                        default = 'rnaseq.tsv',
                        type=str)
    parser.add_argument("-tr","--tracks",
                        help="track paths",
                        nargs = "+",
                        default = ['Mbd3_ESC.bw'],
                        type=str)
    parser.add_argument("-n","--names",
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
    parser.add_argument("-r","--root",
                        help="root folder containing both raw_dir and processed_dir",
                        type = str,
                        default = 'Data/')
    parser.add_argument("-bf", "--buffer",
                        help="Buffer to add each side of a gene to create a graph object",
                        type = int,
                        default = 1e6)
    parser.add_argument("-ch","--chunk_size",
                        help="How many genes to try and process at once",
                        default = 500,
                        type = int
                       )
    parser.add_argument("-chr","--chromosomes",
                        nargs = "+",
                        help="Chromsomes to retrieve",
                        default = CHROMS,
                        type = str)
    args = parser.parse_args()
    
    if isinstance(args.cooler_files,str):
        args.cooler_files = [args.cooler_files]
        
    if isinstance(args.tracks,str):
        args.tracks = [args.tracks]
        
    dset = HiC_Dataset(args.root,
                       contacts=args.cooler_files,
                       tracks=args.tracks,
                       names=args.names,
                       target=args.target,
                       track_transform=mytransform,
                       pre_transform=None,
                       buffer = args.buffer,
                       chunk_size = args.chunk_size,
                       track_statistic_types=args.statistics_types,
                       chromosomes = args.chromosomes
                      )