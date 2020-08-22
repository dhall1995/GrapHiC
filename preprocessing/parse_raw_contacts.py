#!/home/dh486/gnn-env/bin/python
from ..tools.Dataset import pop_HiC_Dataset
from bin_contacts import bin_contacts
from norm_contacts import vc_sqrt_norm

DESCRIPTION = 'Perform uniform region binning of NCC format Hi-C contact data followed by SQRT-VC normalisation as in Rao et al. 2014. Output is stored as an .npz archive which can be used to create the relevant torch graph dataset'

DEFAULT_BIN_SIZE = 50.0
DEFAULT_MIN_BINS = 2
MIN_TRANS_COUNT = 5
DEFAULT_FORMAT = 'NPZ'

if __name__ == "__main__":
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Create a binned and SQRT-VC (Rao et al. 2014) noramlised contact matrix from a .tsv file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-c","--contacts",
                        help="contact file in .tsv format with columns [chrom_a, pos_a, chrom_b, pos_b] (no header)",
                        default = "/home/dh486/rds/hpc-work/Hi_C_GNN/data_raw/contacts/SLX-7671_haploid.tsv")
    parser.add_argument("-o", "--outpath",
                        help="path of the output directory for the numpy archive detailing p-e enhancer links and ABC scores",
                        default = "/home/dh486/rds/hpc-work/Hi_C_GNN/data_torch/raw/",
                        type=str)
    parser.add_argument("-b", "--binsize",
                        help="size of bins to use when binning the contact data, in kb",
                        type = int,
                        default = DEFAULT_BIN_SIZE)
    parser.add_argument("-n", "--normalize",
                        help="Boolean for whether to SQRT-VC normalize the contact data (1) or leave un-normalised (0)",
                        type = int,
                        default = 1)
    arg_parse.add_argument('-m', '--min-bin-count', default=DEFAULT_MIN_BINS, metavar='MIN_BINS', type=int, dest='m',
                         help='The minimum number of bins for chromosomes/contigs; those with fewer than this are excluded from output')

    arg_parse.add_argument('-t', '--min-trans-count', default=MIN_TRANS_COUNT, metavar='MIN_TRANS_COUNT', type=int, dest='t',
                         help='The minimum number contacts for inter-chromosomal contact matrices; those with fewer than this are excluded from output')
    
    args = parser.parse_args()
    
    in_file  = args['c'][0]
    in_file_name = ".".split(os.path.split(in_file)[-1])[0]
    bin_size = args['b']
    if args.normalize:
        out_file_name = "sqrt_vc_normed_" + in_file_name + "_" + str(args.binsize) + "kb"
    else:
        out_file_name = in_file_name + "_" + str(args.binsize) + "kb"
    out_file = os.path.join(args['o'], out_file_name)
    fmt   = DEFAULT_FORMAT # args['f'].upper()  
    min_bins = args['m']
    min_trans = args['t']

    contacts = bin_contacts(in_file, out_file, bin_size, fmt, min_bins, min_trans, save = False)
    
    contacts = {key[0][3:]: contacts[key].tocoo() for key in contacts if key[0][3:] in chroms}
    
    #Need to use SQRT-VC normalisation (Rao et al. 2014 
    #https://www.cell.com/cms/10.1016/j.cell.2014.11.021/attachment/d3c6dcd8-c799-4f68-bbe4-201be54960b5/mmc1 )
    #since we don't have enough coverage for proper KR normalisation
    print("Normalising contact matrices...")
    fn = partial(vc_sqrt_norm,contacts)
    for idx in [1,2,3,4]:
        p = Pool()
        t_outs = p.imap(fn, (key for key in chroms[5*(idx-1):5*idx]))
        for t_out in t_outs:
            key = t_out[0]
            normed_contacts = t_out[1]
            contacts[key] = normed_contacts
        
        p.close()
        p.terminate()
    
    np.savez(out_file,
             **contacts)
    
    #dset = pop_HiC_Dataset("/home/dh486/rds/hpc-work/GNN_Work/data_preprocessed/",
    #                   condition = 'SLX-7671_haploid',
    #                   binSize = args.binsize)