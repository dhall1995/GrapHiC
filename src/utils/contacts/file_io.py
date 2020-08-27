'''
TO DO:
    - Write datatrack_to_bed function
    - Fix datatrack_to_npz function
    - Write bed_to_npz function
''' 

from scipy.sparse import coo_matrix
from scipy import sparse
from ..datatracks.dtrack_utils import non_overlapping, pairRegionsOverlap
#import pandas as pd
import math
import numpy as np
from numpy import int32

CHR_KEY_SEP = ' '

###############################################################################

def load_npz_contacts(file_path, 
        store_sparse=False,
        display_counts=False,
        normalize = False,
        cut_centromeres = True
        ):
    '''
    Utility function to load a .npz file containing contact information from a Hi-C experiment. 
    
    Arguments:
    
    - file_path: A .npz file generated using the nuc_tools ncc_bin tool. The function assumes a
                 File of this format
    - store_sparse: Boolean determining whether to return the contact matrices in sparse format
    - display_counts: Boolean determining whether to display summary plots of Hi-C counts
    - normalize: Boolean determining whether to normalise all matrix elements to lie between zero or one.
                 If False then raw contact counts are returned instead
    - cut_centromeres: Boolean determining whether to cut out the centromeres from the beginning of each
                       chromosome. Since the centromeres contain repetitive elements, they can't currently
                       be mapped by Hi-C so these rows and columns should be void of Hi-C contacts. This 
                       does affect indexing later on but other functions in this package should accommodate
                       for that
                       
    Returns:
    
    - bin_size: The size of each chromatin bin in basepairs.
    - chromo_limits: Dictionary detailing the start and end basepairs of each chromosome
                     in the contact dictionary. NOTE: the chromosome limits are inclusive
                     i.e. for each CHR_A we should have chromo_limits[CHR_A] = (start_A,
                     end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - contacts: Dictionary of matrices detailing contacts between chromosome pairs
    '''
    file_dict = np.load(file_path, allow_pickle=True, encoding = 'bytes')
  
    chromo_limits = {}
    contacts = {}
    bin_size, min_bins = file_dict['params']
    bin_size = int(bin_size*1e3)
  
    chromo_hists = {}
    cis_chromo_hists = {}

    pair_keys = [key for key in file_dict.keys() if "cdata" in key]
    nonpair_keys = [key for key in file_dict.keys() if (CHR_KEY_SEP not in key) and (key != 'params')]
  
    for key in nonpair_keys:
        offset, count = file_dict[key]
        chromo_limits[key] = offset*bin_size, (offset+count)*bin_size
        chromo_hists[key] = np.zeros(count)
        cis_chromo_hists[key] = np.zeros(count)

    maxc = 1
    if normalize:
        for key in sorted(pair_keys):
            maxc = np.maximum(maxc, np.max(file_dict[key]))

    for key in sorted(pair_keys):
        chr_a, chr_b, _ = key.split(CHR_KEY_SEP)
        shape = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "shape"]
        mtype = "CSR"
        try:
            indices = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "ind"]
            indptr = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "indptr"]
        except:
            mtype = "COO"
            row = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "row"]
            col = file_dict[chr_a + CHR_KEY_SEP + chr_b + CHR_KEY_SEP + "col"]

        if mtype == "CSR":

            mat = sparse.csr_matrix((file_dict[key]/maxc, indices, indptr), shape = shape)
        else:
            mat = sparse.coo_matrix((file_dict[key]/maxc, (row, col)), shape = shape)

        if not store_sparse:
            mat = mat.toarray()
          
        if chr_a == chr_b:
            a, b = mat.shape
            cols = np.arange(a-1)
            rows = cols-1

            if not np.all(mat[rows, cols] == mat[cols, rows]): # Not symmetric
                mat += mat.T
          
        contacts[(chr_a, chr_b)] = mat  
     
    #Chromosomes in our dataset
    chroms = chromo_limits.keys()
    if cut_centromeres:
    #Exclude centromeres of chromosomes where we don't have any contact data
        for chrom in chroms:
            chrmax = chromo_limits[chrom][-1]
            temp = contacts[(chrom, chrom)].indices
            chromo_limits[chrom] = (bin_size*np.min(temp[temp>0]), chrmax)
    
    for pair in contacts:
        s0, s1 = int(chromo_limits[pair[0]][0]/bin_size), int(chromo_limits[pair[1]][0]/bin_size)
        try:
            contacts[pair] = contacts[pair][s0:,s1:]
        except:
            contacts[pair] = contacts[pair].tocsr()[s0:, s1:].tocoo()
  
    if display_counts:
        # A simple 1D overview of count densities
 
        from matplotlib import pyplot as plt

        for chr_a, chr_b in contacts:
            mat = contacts[(chr_a, chr_b)]
            chromo_hists[chr_a] += mat.sum(axis=1)
            chromo_hists[chr_b] += mat.sum(axis=0)
 
            if chr_a == chr_b:
                cis_chromo_hists[chr_a] += mat.sum(axis=1)
                cis_chromo_hists[chr_b] += mat.sum(axis=0)
    
        all_sums = np.concatenate([chromo_hists[ch] for ch in chromo_hists])
        cis_sums = np.concatenate([cis_chromo_hists[ch] for ch in chromo_hists])
 
        fig, ax = plt.subplots()
 
        hist, edges = np.histogram(all_sums, bins=25, normed=False, range=(0, 500))
        ax.plot(edges[1:], hist, color='#0080FF', alpha=0.5, label='Whole genome (median=%d)' % np.median(all_sums))

        hist, edges = np.histogram(cis_sums, bins=25, normed=False, range=(0, 500))
        ax.plot(edges[1:], hist, color='#FF4000', alpha=0.5, label='Intra-chromo/contig (median=%d)' % np.median(cis_sums))
 
        ax.set_xlabel('Number of Hi-C RE fragment ends (%d kb region)' % (bin_size/1e3))
        ax.set_ylabel('Count')
 
        ax.legend()
 
        plt.show()

  
    return bin_size, chromo_limits, contacts


##################################################################################

def save_contacts(out_file_path, matrix_dict, chromo_limits, bin_size, min_bins=0):
    '''
    Save Hi-C a Hi-C contact dictionary to a .npz file. 
    
    Arguments:
    
    - out_file_path: Path to save the contact dictionary to
    - matrix_dict: Dictionary of Hi-C contacts. Dictionary should have keys of the form
                   (CHR_A, CHR_B). Trans contact matrices should be stored in sparse COO
                   format while cis contact matrices should be stored in sparse CSR
                   format.
    - chromo_limits: Dictionary detailing the start and end basepairs of each chromosome
                     in the contact dictionary. NOTE: the chromosome limits are inclusive
                     i.e. for each CHR_A we should have chromo_limits[CHR_A] = (start_A,
                     end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - bin_size: What is the size of each contact matrix bin in basepairs e.g. 50000
    - min_bins: Minimum number of bins to be included in a contig (read: chromosome) for it
                to be used downstream. 
    
    '''
    contacts = {}
    kb_bin_size = int(bin_size/1e3)
  
    for chr_a, chr_b in matrix_dict:
        pair = chr_a, chr_b
        key = CHR_KEY_SEP.join(pair)
  
        if chr_a == chr_b:
            contacts[key] = sparse.csr_matrix(matrix_dict[pair])
        else:
            contacts[key] = sparse.coo_matrix(matrix_dict[pair])
    
        start_a, end_a = chromo_limits[chr_a]
        start_b, end_b = chromo_limits[chr_b]
    
        min_a = int(start_a/bin_size)
        num_a = int(math.ceil(end_a/bin_size)) - min_a
        min_b = int(start_b/bin_size)
        num_b = int(math.ceil(end_b/bin_size)) - min_b
    
        # Store bin offsets and spans
        contacts[chr_a] = np.array([min_a, num_a])
        contacts[chr_b] = np.array([min_b, num_b])
    
        contacts['params'] = np.array([kb_bin_size, min_bins])    
  
    np.savez_compressed(out_file_path, **contacts) 

######################################################################
def process_ncc(ncc_file,
                chroms = None,
                contact_region = None
               ):
    if chroms is None:
        chroms = [str(i+1) for i in np.arange(19)] + ['X']
    
    c_dict = {chroms[idx]: {chroms[jdx]: [] for jdx in np.arange(idx, len(chroms))} for idx in np.arange(len(chroms))}
    
    with open(ncc_file, 'r') as ncc:
        for line in ncc:
            linelist = line.split(" ")
            chrom1, start1, end1, chrom2, start2, end2 = linelist[0], linelist[1], linelist[2], linelist[6],linelist[7],linelist[8]
            chrom1 = chrom1[3:]
            chrom2 = chrom2[3:]
            
            start1, start2, end1, end2 = int(start1), int(start2), int(end1), int(end2)
            
            if start1 > end1 and chrom1 == chrom2:
                start1, end1 = end1, start1
            if start2 > end2 and chrom1 == chrom2:
                start2, end2 = end2, start2
                
            pos1 = (0.5)*(int(start1) + int(end1)) 
            pos2 = (0.5)*(int(start2) + int(end2))
            
            if chrom1 not in chroms:
                continue
            if chrom2 not in chroms:
                continue
                
            test_overlaps = non_overlapping(np.array([[start1, end1], [start2, end2]]).astype('int32'))
            if contact_region is not None:
                contact_region = non_overlapping(contact_region)
                test_overlaps = pairRegionsOverlap(test_overlaps, contact_region)
            
            
            if len(test_overlaps) == 0 and chrom1 == chrom2:
                continue
            
                
            if (chrom1 != 'X')&(chrom2 != 'X'):
                chrom1 = int(chrom1)
                chrom2 = int(chrom2)
                if chrom1 > chrom2:
                    chrom1, chrom2 = str(chrom2), str(chrom1)
                    pos1, pos2 = pos2, pos1
            elif chrom1 == 'X':
                chrom1, chrom2 = chrom2, chrom1
                pos1, pos2 = pos2, pos1
            
            chrom1 = str(chrom1)
            chrom2 = str(chrom2)
            c_dict[chrom1][chrom2].append([pos1, pos2])
            
    for chrom1 in c_dict:
        for chrom2 in c_dict[chrom1]:
            c_dict[chrom1][chrom2] = np.array(c_dict[chrom1][chrom2]).astype('int32')
    
    return c_dict