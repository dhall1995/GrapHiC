from src.Datatrack_creation import evaluate_bigwig_over_cooler_bins as eval_bw
import os
ALLOWED_CHROMS = ['chr{}'.format(i+1) for i in range(19)] + ['chrX']

if __name__=="__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Construct feature matrices for all Hi-C bins within a cooler file, provided some bigwig files', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-c","--cooler_file",
                        help="Cooler file from a Hi-C experiment",
                        default = '/disk2/dh486/cooler_files/WT/manually_downsampled/KR/KR_downsampled_WT_merged_10000.cool',
                        type=str)
    parser.add_argument("-b","--bigwigs",
                        help="bigwigs to evaluate",
                        nargs = "+",
                        default = ['/disk2/dh486/bigwigs/GSEXXXXXLaue_CTCF_dipmESC_treat_pileup_filter_norm.bw'],
                        type=str)
    parser.add_argument("-s","--statistics_types",
                        help="bigwigs to evaluate",
                        nargs = "+",
                        default = 'max',
                        type=str)
    parser.add_argument("-o","--out_path",
                        help="Path to save dataframe to",
                        type = str,
                        default = None)
    args = parser.parse_args()
    
    if args.out_path is None:
        clrname = os.path.split(args.cooler_file)[-1].split(".")[0]
        args.out_path = clrname + "_bw_data.tsv"
    
    df = eval_bw(args.cooler_file,
                 bwpaths = args.bigwigs,
                 stats_types = args.statistics_types,
                 allowed_chroms = ALLOWED_CHROMS
                )
    
    df.to_csv(args.out_path, sep = "\t")
    
    

    