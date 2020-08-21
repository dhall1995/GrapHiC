from scipy.sparse import coo_matrix
from utils.file_io import load_npz_contacts as lnc
from ..misc import one_hot, bp_to_idx, idx_to_bp


from scipy.interpolate import CubicSpline
import torch
from torch_geometric.utils import is_undirected
import numpy as np
import h5py as h
import pandas as pd
import os.path as osp

from multiprocessing import Pool
from functools import partial


def get_cont_distance(chr_lims, binSize, chroms, const, mycont):
    '''
    Given a some chromosome limits, binSizes and which chromosomes we are interested in, returns the log10 
    contact distance between two indexes, cont. If the two indexes encode bins within different chromosomes 
    then the pair is given a score of -5.
    
    Arguments:
    
    - chr_lims: Dictionary detailing the start and end basepairs of each chromosome in the contact dictionary.
                NOTE: the chromosome limits are inclusive i.e. for each CHR_A we should have
                chromo_limits[CHR_A] = (start_A,end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - binSize: The length (in basepairs) of each chromatin bin in our contact map.
    - chroms: A list of chromosomes in our Hi-C map. 
    - const: Constant to return if indexes in cont correspond to bins on different chromosomes.
    - cont: A 2-tuple of indexes from our contact map. i.e. inputting (i,j) would output the log10 backbone 
            distance between i and j if those indices correspond to bins on the same chromosome or just a
            constant if they correspond to bins on different chromosomes 
         
    
    Returns:
    
    - log10dist: log10 of the backbone distance between those Hi-C indiexes.
    
    
    '''  
    
    idx, cont = mycont[0], mycont[1]
    chrom, chrbp = idx_to_bp(cont[0], chr_lims , binSize, chroms)
    chrom2, chrbp2 = idx_to_bp(cont[1], chr_lims , binSize, chroms)
    
    if (chrom2 == chrom):
        if abs(chrbp2-chrbp) > 0:
            return idx, np.log10(abs(chrbp2-chrbp))
        else:
            return idx, 0
    else:
        return idx, const

def backbone_restraint(row, col, size, index):
    '''
    Given a row and column vectors detailing a cis-Hi-C contact matrix (essentially in in COO format) as well
    as an index, checks whether that index has Hi-C contacts with its adjascent indices along the backbone. 
    
    Arguments:
    
    - row: row indexes from COO format sparse matrix. Assumed to be zero indexed
    - col: column indexes from COO format sparse matrix. Assumed to be zero indexed
    - index: index to be checked     
    
    Returns:
    
    - pos: Bool. If true, then the contact [index, index +1] is in the matrix
    - neg: Bool. If ture, then the contact [index-1, index] is in the matrix
    
    '''  
    
    #Assume backbone restraint satisfied
    pos = True
    neg = True
    
    if index < size-1:
        #if index is at the end of the chr so postive strand restraint already satisfied
        #is index in rows?
        row_idx = row == index
        if np.sum(row_idx) == 0:
            #index not in the rows
            pos = False
        else:
            row_idx_cols = col[row_idx]
            if index+1 not in row_idx_cols:
                pos = False
                
    if index > 0:
        #if index is zero then the negative strand restraint is automatically satisfied
        #Is index in the cols?
        col_idx = col == index
        if np.sum(col_idx) == 0:
            #index not in the cols
            neg = False
        else:
            col_idx_rows = row[col_idx]
            if index-1 not in col_idx_rows:
                neg = False
    
    return pos, neg, index
    
    
    

def make_ptg_pophic_edge_index(file, verbose = True):
    '''
    Creates a host of relevant objects involved in the construction of a pytorch geometric population Hi-C
    edge index object detailing graph edges and strengths. 
    
    Arguments:
    
    - file: .npz file generated using the ncc_bin tool from nuc_tools (see https://github.com/tjs23/nuc_tools)
    - verbose: Bool. Whether to print out progress updates.
    
    Returns:
    
    - shape: resulting shape of the contact matrix
    - binSize: The length (in basepairs) of each chromatin bin in our contact map. 
    - chr_lims: Dictionary detailing the start and end basepairs of each chromosome in the contact dictionary.
                NOTE: the chromosome limits are inclusive i.e. for each CHR_A we should have
                chromo_limits[CHR_A] = (start_A,end_A) where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
    - egde_index: Edge index matrix of the contact map. This is essentially the contact map in COO format
                  where we only detail which nodes are in contact with which other nodes
    - edge_attr: Edge attribute matrix. Here we detail the edge strength of each edge in the conact matrix
    
    
    '''    
    binSize,y,z = lnc(file, store_sparse = True)
    
    
    order = ['chr' + str(i+1) for i in np.arange(19)] + ['chrX']

    offsets = np.cumsum([int((y[i][1]-y[i][0])/binSize) for i in order])
    shape = (offsets[-1], offsets[-1])
    lenchr1 = offsets[0] 
    offsets -= lenchr1
    offsets = {chrom: offsets[i] for i, chrom in enumerate(order)}
    
    myrows = np.empty((1))
    mycols = np.empty((1))
    mydata = np.empty((1))
    for item in z.keys():
        z[item] = z[item].tocoo()
        
        
        chr1 = item[0]
        chr2 = item[1]
        if item[0] != 'chrY' and item[1] != 'chrY':
            rowadd = z[item].row
            coladd = z[item].col
            dataadd = z[item].data
            
            #If same chromosome then check backbone edges
            if item[0] == item[1]:
                backbone_idxs = abs(rowadd - coladd) == 1
                if verbose:
                    print("Checking backbone restraints on chromosome {}".format(item[0]))
                lenchrom = z[item].shape[0]
                if verbose:
                    print("Chromosome length: {}".format(lenchrom))
                fn = partial(backbone_restraint, rowadd[backbone_idxs], coladd[backbone_idxs],lenchrom)
                p = Pool()
                temp_outputs = p.imap(fn, (idx for idx in range(lenchrom)))
                
                postot = 0
                negtot = 0
                for temp_output in temp_outputs:
                    if not temp_output[0]:
                        rowadd = np.append(rowadd, temp_output[2])
                        coladd = np.append(coladd, temp_output[2] + 1)
                        dataadd = np.append(dataadd, 1)
                        postot += 1
                    if not temp_output[1]:
                        rowadd = np.append(rowadd, temp_output[2]-1)
                        coladd = np.append(coladd, temp_output[2])
                        dataadd = np.append(dataadd, 1)
                        negtot += 1
               
                if verbose:
                    print("Added in {} positive strand and {} negative strand backbone contacts for {}".format(postot, negtot, item[0]))
                    print("Done!")
                
            rowadd = rowadd + offsets[chr1]
            coladd = coladd + offsets[chr2]
            
            myrows = np.append(myrows, rowadd).astype(int)
            mycols = np.append(mycols, coladd).astype(int)
            mydata = np.append(mydata, dataadd)
            
            if item[1] != item[0]:
                myrows = np.append(myrows, coladd).astype(int)
                mycols = np.append(mycols, rowadd).astype(int)
                mydata = np.append(mydata, dataadd)
    
    del y['chrY']
    chr_lims = {chrom[3:]: y[chrom] for chrom in y}
    
    temp_coo = coo_matrix((mydata, (myrows, mycols)), shape = shape)
    
    edge_index = np.append(temp_coo.row[None,:], temp_coo.col[None,:], axis = 0)
    edge_attr = temp_coo.data[None,:]

    #add in backbone edges and give each edge an attribute determining whether it is
    #trans or distal cis. This will be the log-length of the edge size for cis contacts
    # and -5 for trans contacts
    chroms = [str(i+1) for i in np.arange(19)] + ['X']
    
    fn = partial(get_cont_distance, chr_lims, binSize, chroms, -5)
    p = Pool()
    lengths = np.zeros(edge_attr.shape)
    
    if verbose:
        print("Adding in edge lengths...")
    temp_outputs = p.imap(fn, ((idx, edge_index[:,idx]) for idx in range(edge_index.shape[1])))
    for temp_output in temp_outputs:
        lengths[0,temp_output[0]] = temp_output[1]
    
    if verbose:
        print("Done!")
    edge_attr = np.append(edge_attr, lengths, axis = 0)

    return  shape, binSize, chr_lims, edge_index, edge_attr