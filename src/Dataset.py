import os.path as osp
import itertools
import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import Pool

import torch
from torch_geometric.data import Dataset

from .Datatrack_creation import evaluate_bigwigs_over_cooler_bins as eval_bws
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


def split_data(df):
    prom_regions = {k1: np.concatenate([item[None,1:3] for item in list(g1)],
                                       axis = 0).astype('int32') for k1,g1 in itertools.groupby(sorted(rnaseq.values,
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
    graph_regions,
    names,
    target,
    binned_data,
    prom_data,
    coolers,
    chrom,
    out_path,
    pbar
):
    strchrom = name_chr(chrom)
    shape = prom_regions[chrom].shape[0]
    
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
                       os.path.join(out_path,"data_{}_{}".format(strchrom, 
                                                                 ticker
                                                                )
                                   )
                      )
        pbar.update(1)
        
        
def make_chromo_gene_graphs(
    clrs,
    graph_regions, 
    names,
    target,
    binned_data,
    prom_data,
    chunk_size,
    out_path,
    chroms,
    chrom
):
    
    chrom_pos = np.array([idx for idx, item in enumerate(chroms) if item == chrom])[0]
    pbar = tqdm(total=shape, 
                position=chrom_pos,
                desc = chrom)
    for idx in np.arange(0,
                         shape,
                         chunk_size):
        make_chunk_gene_graphs(
            idx, 
            chunk_size,
            graph_regions,
            names,
            target,
            binned_data,
            prom_data,
            clrs,
            chrom,
            out_path,
            pbar
        )
    
    return None

class HiC_Dataset(Dataset):
    def __init__(self, 
                 root,
                 contacts=None,
                 bigwigs=None,
                 names=None,
                 target=None,
                 transform=None, 
                 pre_transform=None,
                 buffer = 25e4,
                 chunk_size = 500,
                 bw_statistic_types=['mean'],
                 chromosomes = CHROMS
                ):
        super(MyOwnDataset, self).__init__(root, 
                                           transform, 
                                           pre_transform)
        self.contacts = contacts
        c = cooler.Cooler(self.contacts[0])
        self.chr_lims = {str(row[0]):int(row[1]) for row in c.chroms()[:].values}
        
        if self.names is None:
            self.bigwigs = [bigwigs[key] for key in self.bigwigs]
            self.names = [key for key in self.bigwigs]
        else:
            self.bigwigs = [bigwigs[key] for key in self.names]
        self.target = target
        self.stats_types = bw_statistic_types
        self.buffer = buffer
        self.chunk_size = chunk_size
        self.chromosomes = chromosomes
        
        self.num_objects = pd.read_table(self.target).values.shape[0]
        
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
        #EVALUATE BIGWIGS OVER COOLER BINS AND SAVE TO FILE
        df = eval_bws_over_cooler(self.contacts[0],
                                  bwpaths = self.bigwigs,
                                  names = self.names,
                                  stats_types = self.stats_types,
                                  allowed_chroms = ALLOWED_CHROMS
                                 )
        
        if self.transform == "Power":
            norm = PowerTransform_norm
        elif self.transform == "Standard":
            norm = Standard_norm
        elif self.transform == "Robust":
            norm = Robust_norm
    
        if self.transform is not None:
            df = pd.DataFrame(data= norm(df.values), 
                          columns = df.columns,
                          index= df.index)
        
        df.to_csv(os.path.join(self.processed_dir, "cooler_bigwig_data.csv"), 
                  sep = "\t")
        
        #EVALUATE BIGWIGS OVER THE TARGETS AND SAVE TO FILE
        df = pd.read_table(self.target)
        df = df[df.columns.values[:5]]
        arr, colnames = eval_bws_over_bed_df(df,
                                          self.bigwigs,
                                          names = self.names,
                                          stats_types = self.stats_types)
        arr = norm(arr)
        for idx,name in enumerate(colnames):
            df[name] = arr[:,idx]
        
        df.to_csv(self.target,
                  sep="\t",
                  index=False)
        
        df = df.loc[[item in self.chromosomes for item in df['chromosome']]]
        
        #MAKE GENE GRAPHS
        c = cooler.Cooler(self.contacts[0])
        chr_lims = {str(row[0]):int(row[1]) for row in c.chroms()[:].values}
        
        prom_regions, target, names, pdata = split_df(df)

        graph_regions = {chrom: buffer_regs(prom_regions[chrom],
                                            lims = [0,self.chr_lims[chrom]],
                                            buff = self.buffer
                                           ) for chrom in prom_regions}
    
    
        fn = partial(make_chromo_gene_graphs, 
                     self.contacts,
                     graph_regions,
                     names,
                     target,
                     os.path.join(self.processed_dir, "cooler_bigwig_data.csv"),
                     pdata,
                     self.chunk_size,
                     self.processed_dir,
                     [item for item in self.chromosomes if item in graph_regions]
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