from ..utils.contacts.file_io import load_npz_contacts as lnc
import numpy as np
from multiprocessing import Pool
from functools import partial
from scipy.sparse import coo_matrix as coo
import os.path as osp

chroms = [str(i+1) for i in np.arange(19)] + ['X']

def vc_sqrt_norm(contacts, key):
    print('Starting chromosome {}'.format(key))
    sums = np.sum(contacts[key], axis = 0)
    norms = np.dot(sums.T, sums)
    goodcoverage = np.where(np.array(sums>0)[0,:])[0]
    norms[np.ix_(goodcoverage,goodcoverage)] = np.sqrt(1./norms[np.ix_(goodcoverage,goodcoverage)])
    
    newdata = np.multiply(contacts[key].data, [norms[contacts[key].row[idx],
                                                     contacts[key].col[idx]] for idx in np.arange(contacts[key].row.shape[0])])


    mat = coo((newdata, (contacts[key].row, contacts[key].col)), shape=contacts[key].shape)
    print('Done chromosome {}'.format(key))
    return key, mat

if __name__ == "__main__":
    import sys
    import argparse
    parser = argparse.ArgumentParser(description='Read in a file or set of files, and return the result.', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument("-c","--contacts",
                        help="contact file in .npz format generated using nuc_tools bin_csv function",
                        default = "data/raw/contacts/SLX-7671_haploid_5kb.npz")
    parser.add_argument("-o", "--outpath",
                        help="path of the output directory for the numpy archive detailing p-e enhancer links and ABC scores",
                        default = "data/processed/contacts",
                        type=str)
    
    args = parser.parse_args()
    
    binsize, chr_lims, contacts = lnc(args.contacts,store_sparse=True,
                                      display_counts=False,
                                      normalize = False,
                                      cut_centromeres = False,
                                      cis = True)
    
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
    
    np.savez(osp.join(args.outpath,
                      "sqrt_normed_{}".format(osp.split(args.contacts)[-1][:-4])
                         ),
             **contacts
            )
