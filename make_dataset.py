from src.Dataset import HiC_Dataset

if __name__=="__main__":
    
    parser = argparse.ArgumentParser(description='Construct a datset of per-gene graph objects based off contact info, some target (e.g. rnaseq) and protein binding data', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c","--cooler_files",
                        nargs = "+",
                        help="Cooler file from a Hi-C experiment",
                        default = 'KR_downsampled_WT_merged_10000.cool',
                        type=str)
    parser.add_argument("-t","--target",
                        help="path tsv with columns ['chrom','promoter_start','promoter_end','target','name']",
                        default = 'rnaseq.tsv',
                        type=str)
    parser.add_argument("-b","--bigwigs",
                        help="bigwig file names",
                        nargs = "+",
                        default = ['Mbd3_ESC.bw'],
                        type=str)
    parser.add_argument("-n","--names",
                        help="names of each datatrack",
                        nargs = "+",
                        default = None,
                        type=str)
    parser.add_argument("-s","--statistics_types",
                        help="bigwigs to evaluate",
                        nargs = "+",
                        default = 'max',
                        type=str)
    parser.add_argument("-tf","--transform",
                        help="Transform to apply to the data",
                        default = None)
    parser.add_argument("-r","--root",
                        help="root folder containing both raw_dir and processed_dir",
                        type = str,
                        default = 'Data/')
    parser.add_argument("-bf", "--buffer",
                        help="Buffer to add each side of a gene to create a graph object",
                        type = int,
                        default = 25e4)
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
        
    if isinstance(args.bigwigs,str):
        args.bigwigs = [args.bigwigs]
        
    dset = HiC_Dataset(self,
                       args.root,
                       contacts=args.contacts,
                       bigwigs=args.bigwigs,
                       names=args.names,
                       target=args.target,
                       transform=args.transform,
                       pre_transform=None,
                       buffer = args.buffer,
                       chunk_size = args.chunk_size,
                       bw_statistic_types=args.statistics_types,
                       chromosomes = args.chromosomes
                      )