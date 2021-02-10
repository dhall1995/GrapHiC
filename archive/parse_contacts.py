#!/home/dh486/gnn-env/bin/python
from src.utils.contacts.bin_contacts import bin_contacts
from src.utils.contacts.sqrt_norm import vc_sqrt_norm, row_col_sums
from src.utils.contacts.file_io import load_npz_contacts as lnc
import numpy as np
from functools import partial
from multiprocessing import Pool


DESCRIPTION = 'Perform uniform region binning of NCC format Hi-C contact data followed by SQRT-VC normalisation as in Rao et al. 2014. Output is stored as an .npz archive which can be used to create the relevant torch graph dataset'

DEFAULT_BIN_SIZE = 50
DEFAULT_MIN_BINS = 2
MIN_TRANS_COUNT = 5
DEFAULT_FORMAT = 'NPZ'
CHROMS = [str(i+1) for i in np.arange(19)] + ['X']
CHR_KEY_SEP = " "

if __name__ == "__main__":
    
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description='Create a binned and SQRT-VC (Rao et al. 2014) noramlised contact matrix from a .tsv file', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-c","--contacts",
                        help="contact file in .tsv format with columns [chrom_a, pos_a, chrom_b, pos_b] (no header)",
                        default = "/home/dh486/rds/hpc-work/Hi_C_GNN_data/data_raw/contacts/SLX-7671_haploid.tsv")
    parser.add_argument("-o", "--outpath",
                        help="path of the output directory for the numpy archive detailing normalised contacts",
                        default = "/home/dh486/rds/hpc-work/Hi_C_GNN_data/data_torch/raw/",
                        type=str)
    parser.add_argument("-b", "--binsize",
                        help="size of bins to use when binning the contact data, in kb",
                        type = int,
                        default = DEFAULT_BIN_SIZE)
    parser.add_argument("-n", "--normalize",
                        help="Boolean for whether to SQRT-VC normalize the contact data (1) or leave un-normalised (0)",
                        type = int,
                        default = 1)
    parser.add_argument('-m', '--minbincount', default=DEFAULT_MIN_BINS, type=int,
                         help='The minimum number of bins for chromosomes/contigs; those with fewer than this are excluded from output')

    parser.add_argument('-t', '--mintranscount', default=MIN_TRANS_COUNT, type=int,
                         help='The minimum number contacts for inter-chromosomal contact matrices; those with fewer than this are excluded from output')
    
    args = parser.parse_args()
    
    in_file  = args.contacts
    
    in_file_name = (os.path.split(in_file)[1]).split(".")[0]
    bin_size = args.binsize
    if args.normalize:
        out_file_name = "sqrt_vc_normed_" + in_file_name + "_" + str(args.binsize) + "kb"
    else:
        out_file_name = in_file_name + "_" + str(int(args.binsize)) + "kb" 
    out_file = os.path.join(args.outpath, out_file_name)
    fmt   = DEFAULT_FORMAT  
    min_bins = args.minbincount
    min_trans = args.mintranscount

    print("Binning contacts...")
    contacts = bin_contacts(in_file, out_file + ".npz", bin_size, fmt, min_bins, min_trans)
    

    if args.normalize == 1:
        print("Loading unnormalised contact matrices...")
        binsize, chr_lims, contacts = lnc(out_file + ".npz",
                                      store_sparse=True,
                                      display_counts=False,
                                      normalize = False,
                                      cut_centromeres = False)
    
        contacts = {(key[0][3:],key[1][3:]): contacts[key].tocoo() for key in contacts if key[0][3:] in CHROMS and key[1][3:] in CHROMS}
        print("working out row sums...")
        sums = row_col_sums(contacts, chroms = CHROMS)
        #Need to use SQRT-VC normalisation (Rao et al. 2014 
        #https://www.cell.com/cms/10.1016/j.cell.2014.11.021/attachment/d3c6dcd8-c799-4f68-bbe4-201be54960b5/mmc1 )
        #since we don't have enough coverage for proper KR normalisation
        print("Normalising contact matrices...")
        fn = partial(vc_sqrt_norm,contacts, sums)
        p = Pool()
        t_outs = p.imap(fn, (key for key in contacts.keys()))
        for t_out in t_outs:
            key = t_out[0]
            normed_contacts = t_out[1]
            contacts[key] = normed_contacts
        
        p.close()
        p.terminate()
    
        out = {}
        for chromo_key in contacts:
            chr_a, chr_b = chromo_key
            chr_a = "chr" + chr_a
            chr_b = "chr" + chr_b
            data_key = chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "cdata"    
      
            if chr_a == chr_b:
                indices_key = chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "ind"
                indptr_key = chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "indptr"
                shape_key = chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "shape"

                cdata = contacts[key].tocsr()
                out[data_key] = cdata.data.astype('float')
                out[indices_key] = cdata.indices.astype('int32')
                out[indptr_key] = cdata.indptr.astype('int32')
                out[shape_key] = cdata.shape

            else:
                row_key = chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "row"
                col_key = chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "col"
                shape_key = chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "shape"

                cdata = contacts[key]
                out[data_key] = cdata.data.astype('float')
                out[row_key] = cdata.row.astype('int32')
                out[col_key] = cdata.col.astype('int32')
                out[shape_key] = cdata.shape
        
            # Store bin offsets and spans
            out[chr_a] = np.array([0, cdata.shape[0]])
            out[chr_b] = np.array([0, cdata.shape[1]])
    
        out['params'] = np.array([binsize, args.minbincount])
        
     
        np.savez_compressed(out_file, **out)
    
    print("Done.")
    