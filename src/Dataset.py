import os
import itertools
import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import Pool
import cooler

import torch
from torch_geometric.data import Dataset

from .Datatrack_creation import evaluate_bigwigs_over_cooler_bins as eval_bws_over_cooler
from .Datatrack_creation import evaluate_bigwigs_over_bed_dataframe as eval_bws_over_bed_df
from .utils.Datatrack import DataTrack_bigwig as dtbw
from .utils.norm import (
    PowerTransform_norm, 
    Standard_norm, 
    Robust_norm
)
from .utils.misc import (
    buffer_regs, 
    ptg_from_npy, 
    name_chr
)
from .Graph_creation import (
    compute_ptg_graph_from_regions,
    add_binned_data_to_graphlist
)

CHROMS = [f'chr{idx+1}' for idx in np.arange(19)]+['chrX']

def split_data(df):
    prom_regions = {k1: np.concatenate([item[None,1:3] for item in list(g1)],
                                       axis = 0).astype('int32') for k1,g1 in itertools.groupby(sorted(df.values,
                                                                                                       key = lambda x:x[0]),
                                                                                                lambda x: x[0])}

    target = {k1: np.concatenate([item[None,[3]] for item in list(g1)],
                                 axis = 0).astype('float')[:,0] for k1,g1 in itertools.groupby(sorted(df.values,
                                                                                                      key = lambda x:x[0]),
                                                                                               lambda x: x[0])}
    names = {k1: np.concatenate([item[None,[4]] for item in list(g1)],
                                axis = 0)[:,0] for k1,g1 in itertools.groupby(sorted(df.values,
                                                                                     key = lambda x:x[0]),
                                                                              lambda x: x[0])}
    
    pdata = {k1: np.concatenate([item[None,5:] for item in list(g1)],
                                axis = 0).astype('float') for k1,g1 in itertools.groupby(sorted(df.values,
                                                                                                key = lambda x:x[0]),
                                                                                         lambda x: x[0])}
    
    return prom_regions, target, names, pdata

def make_chunk_gene_graphs(
    idx,
    chunk_size,
    shape,
    graph_regions,
    names,
    target,
    binned_data,
    prom_data,
    coolers,
    chrom,
    out_path
):
    if idx+chunk_size > shape:
        lim = shape
    else:
        lim = idx+chunk_size
    rdict = {chrom: graph_regions[chrom][idx:lim,:]}
    ndict = {chrom: names[chrom][idx:lim]}
    tdict = {chrom: target[chrom][idx:lim]}
    prom_info = {chrom: prom_data[chrom][idx:lim,:]}
    glist = compute_ptg_graph_from_regions(coolers,
                                           rdict,
                                           names = ndict,
                                           balance = True,
                                           join = False,
                                           force_disjoint=False,
                                           record_cistrans_interactions = False,
                                           record_node_chromosome_as_onehot = False)
    add_binned_data_to_graphlist(glist[chrom],
                                 binned_data)
    
    for jdx,item in enumerate(glist[chrom]):
        item['target'] = tdict[chrom][jdx]
        item['prom_x'] = prom_info[chrom][jdx,:]
        torch_item = ptg_from_npy(item)
        torch.save(torch_item, 
                   os.path.join(out_path,"data_{}_{}".format(name_chr(chrom), 
                                                             jdx+idx
                                                            )
                               )
                  )
        
        
def make_chromo_gene_graphs(
    clrs,
    graph_regions, 
    names,
    target,
    binned_data,
    prom_data,
    chunk_size,
    out_path,
    chrom
):
    
    shape = graph_regions[chrom].shape[0]
    print(f"\tstarting {chrom} with {shape} regions of interest")
    for idx in np.arange(0,
                         shape,
                         chunk_size):
        if idx+chunk_size > shape:
            lim = shape
        else:
            lim = idx+chunk_size
        rdict = {chrom: graph_regions[chrom][idx:lim,:]}
        ndict = {chrom: names[chrom][idx:lim]}
        tdict = {chrom: target[chrom][idx:lim]}
        prom_info = {chrom: prom_data[chrom][idx:lim,:]}
        glist = compute_ptg_graph_from_regions(clrs,
                                           rdict,
                                           names = ndict,
                                           balance = True,
                                           join = False,
                                           force_disjoint=False,
                                           record_cistrans_interactions = False,
                                           record_node_chromosome_as_onehot = False)
        add_binned_data_to_graphlist(glist[chrom],
                                     binned_data)
    
        for jdx,item in enumerate(glist[chrom]):
            item['target'] = tdict[chrom][jdx]
            item['prom_x'] = prom_info[chrom][jdx,:]
            torch_item = ptg_from_npy(item)
            torch.save(torch_item, 
                       os.path.join(out_path,"data_{}_{}.pt".format(name_chr(chrom), 
                                                                 jdx+idx
                                                                )
                                   )
                      )

    print(f"\tfinished {chrom}")
    return None

class HiC_Dataset(Dataset):
    def __init__(self, 
                 root,
                 contacts=None,
                 bigwigs=None,
                 names=None,
                 target=None,
                 bw_transform=None,
                 transform=None,
                 pre_transform=None,
                 buffer = 25e4,
                 chunk_size = 500,
                 bw_statistic_types=['mean'],
                 chromosomes = CHROMS
                ):
        #Estimate number of objects by rows of our target csv
        self.target = target
        self.num_objects = pd.read_table(os.path.join(root,
                                                      os.path.join('raw',
                                                                   self.target)
                                                     )
                                        ).values.shape[0]
        
        #Initialise class specific attributes
        self.contacts = contacts
        c = cooler.Cooler(os.path.join(root,
                                       os.path.join('raw/contacts',
                                                    self.contacts[0]
                                                   )
                                      )
                         )
        self.chr_lims = {str(row[0]):int(row[1]) for row in c.chroms()[:].values}
        
        if names is None:
            self.names = bigwigs
        else:
            self.names = names
            
        self.bigwigs = bigwigs    
        self.stats_types = bw_statistic_types
        self.bw_transform = bw_transform
        self.buffer = buffer
        self.chunk_size = chunk_size
        self.chromosomes = chromosomes
        
        #Initialise the class from the parent intialisation
        super(HiC_Dataset, self).__init__(root,
                                          transform,
                                          pre_transform)
                
    @property
    def raw_file_names(self):
        conts = [os.path.join('contacts',item) for item in self.contacts]
        bigwigs = [os.path.join('bigwigs',item) for item in self.contacts]
        return [self.contacts, 
                self.bigwigs, 
                self.target]

    @property
    def processed_file_names(self):
        return [f'data_{idx}.pt' for idx in np.arange(self.num_objects)]

    def process(self):
        full_contact_paths= [os.path.join(self.root,
                                          os.path.join('raw/contacts',
                                                       path
                                                      )
                                         ) for path in self.contacts]
        full_bigwig_paths= [os.path.join(self.root,
                                         os.path.join('raw/bigwigs',
                                                      path
                                                     )
                                        ) for path in self.bigwigs]
        full_target_path = os.path.join(self.root,
                                        os.path.join('raw',
                                                     self.target
                                                    )
                                       )
        #EVALUATE BIGWIGS OVER COOLER BINS AND SAVE TO FILE
        bw_out_file = os.path.join(self.processed_dir, "cooler_bigwig_data.csv")
        print(f"Evaluating bigwigs over cooler bins and saving to file {bw_out_file}")
        df = eval_bws_over_cooler(full_contact_paths[0],
                                  bwpaths = full_bigwig_paths,
                                  names = self.names,
                                  stats_types = self.stats_types,
                                  allowed_chroms = self.chromosomes
                                 )
        
        if self.bw_transform == "Power":
            norm = PowerTransform_norm
        elif self.bw_transform == "Standard":
            norm = Standard_norm
        elif self.bw_transform == "Robust":
            norm = Robust_norm
    
        if self.bw_transform is not None:
            df = pd.DataFrame(data= norm(df.values),
                              columns = df.columns,
                              index= df.index)
        
        df.to_csv(bw_out_file, 
                  sep = "\t")
        
        #EVALUATE BIGWIGS OVER THE TARGETS AND SAVE TO FILE
        print(f"Evaluating bigwigs over specific regions of interest and appending to target file {full_target_path}")
        df = pd.read_table(full_target_path)
        df = df[df.columns.values[:5]]
        colnames, arr = eval_bws_over_bed_df(df,
                                             full_bigwig_paths,
                                             names = self.names,
                                             stats_types = self.stats_types)
        if self.bw_transform is not None: 
            arr = norm(arr)
        for idx,name in enumerate(colnames):
            df[name] = arr[:,idx]
        
        df.to_csv(full_target_path,
                  sep="\t",
                  index=False)
        
        df = df.loc[[item in self.chromosomes for item in df['chromosome']]]
        
        #MAKE GENE GRAPHS
        print(f"Extracting contact info and constructing Hi-C graphs for {self.num_objects} regions of interest")
        prom_regions, target, target_names, pdata = split_data(df)

        graph_regions = {chrom: buffer_regs(prom_regions[chrom],
                                            lims = [0,self.chr_lims[chrom]],
                                            buff = self.buffer
                                           ) for chrom in prom_regions}
    
        fn = partial(make_chromo_gene_graphs, 
                     full_contact_paths,
                     graph_regions,
                     target_names,
                     target,
                     bw_out_file,
                     pdata,
                     self.chunk_size,
                     self.processed_dir,
                )
        p = Pool()
        t_outs = p.imap(fn, (chrom for chrom in graph_regions))
        for t_out in t_outs:
            pass
            
        print("Renaming files")
        jdx = 0
        for idx, file in enumerate(glob.glob(os.path.join(self.processed_dir, 
                                                          "*.pt"))):
            os.rename(file, 
                      os.path.join(self.processed_dir,
                                   f"data_{idx}.pt")
                     )
            jdx = idx

        self.num_objects = jdx
        
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data