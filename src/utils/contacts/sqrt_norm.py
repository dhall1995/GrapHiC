from .file_io import load_npz_contacts as lnc
import numpy as np
from multiprocessing import Pool
from functools import partial
from scipy.sparse import coo_matrix as coo
import os.path as osp

chroms = [str(i+1) for i in np.arange(19)] + ['X']

def row_col_sums(contacts, chroms = chroms):
    chromshapes = {}
    for chrom in chroms:
        chromshapes[chrom] = contacts[(chrom,chrom)].shape[0]
        
    sums = {chrom: np.zeros((chromshapes[chrom],1)) for chrom in chroms}
    
    for key in contacts:
        c1,c2 = key
        if c1 not in chroms or c2 not in chroms:
            continue
        sums[c1] += np.sum(contacts[key],axis =1)
        sums[c2] += np.sum(contacts[key],axis = 0).T 

    return sums

def vc_sqrt_norm(contacts, sums, key):
    c1,c2 = key
    print('Starting chromosome pair {}-{}'.format(c1,c2))
    norms = np.dot(sums[c1], sums[c2].T)
    #goodcoverage = {c: np.where(np.array(sums[c]>0)[0,:])[0] for c in [c1,c2]}
    #norms[np.ix_(goodcoverage[c1],goodcoverage[c2])] = np.sqrt(1./norms[np.ix_(goodcoverage[c1],goodcoverage[c2])])
    norms[norms>0] = np.sqrt(1./norms[norms>0])
    
    newdata = np.multiply(contacts[key].data, [norms[contacts[key].row[idx],
                                                     contacts[key].col[idx]] for idx in np.arange(contacts[key].row.shape[0])])

    
    mat = coo((newdata, (contacts[key].row, contacts[key].col)), shape=contacts[key].shape)
    print('Done chromosome {}'.format(key))
    return key, mat

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='sqrt normalise contacts', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-c","--contacts",
                        help="contact file in .npz format generated using nuc_tools bin_csv function",
                        default = "/home/dh486/rds/hpc-work/Hi_C_GNN_data/data_raw/contacts/SLX-7671_haploid_50kb.npz")
    parser.add_argument("-o", "--outpath",
                        help="path of the output directory for the numpy archive detailing p-e enhancer links and ABC scores",
                        default = "/home/dh486/rds/hpc-work/Hi_C_GNN_data/data_torch/raw/",
                        type=str)
    
    args = parser.parse_args()
    
    binsize, chr_lims, contacts = lnc(args.contacts,store_sparse=True,
                                      display_counts=False,
                                      normalize = False,
                                      cut_centromeres = False)
    
    contacts = {(key[0][3:],key[1][3:]): contacts[key].tocoo() for key in contacts if key[0][3:] in chroms and key[1][3:] in chroms}
    
    sums = row_col_sums(contacts, chroms = chroms)
    #Need to use SQRT-VC normalisation (Rao et al. 2014 
    #https://www.cell.com/cms/10.1016/j.cell.2014.11.021/attachment/d3c6dcd8-c799-4f68-bbe4-201be54960b5/mmc1 )
    #since we don't have enough coverage for proper KR normalisation
    print("Normalising contact matrices...")
    fn = partial(vc_sqrt_norm,contacts,sums)
    p = Pool()
    t_outs = p.imap(fn, (key for key in contact.keys()))
    for t_out in t_outs:
        key = t_out[0]
        normed_contacts = t_out[1]
        contacts[key] = normed_contacts
        
    p.close()
    p.terminate()
    
    np.savez(osp.join(args.outpath,
                      "sqrt_normed_{}".format(osp.split(args.contacts)[-1][:-4])
                         ),
             **contacts
            )
