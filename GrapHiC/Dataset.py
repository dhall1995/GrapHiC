import os
import glob
import itertools
import pandas as pd
import numpy as np
from functools import partial
from multiprocessing import Pool
import cooler

from sklearn.model_selection import train_test_split as tts

import torch
from torch_geometric.data import Dataset

from .Datatrack_creation import evaluate_tracks_over_bed_dataframe as eval_tracks_over_bed_df
from .Datatrack_creation import evaluate_tracks_over_cooler_bins as eval_tracks_over_cooler
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
    from_regions,
    add_binned_data_to_graphlist
)

CHROMS = [f'chr{idx+1}' for idx in np.arange(19)]+['chrX']

def split_data(df):
    namecol = np.where(df.columns.values=='name')[0][0]
    
    regions = {k1: np.concatenate([item[None,1:3] for item in list(g1)],
                                       axis = 0).astype('int32') for k1,g1 in itertools.groupby(sorted(df.values,
                                                                                                       key = lambda x:x[0]),
                                                                                                lambda x: x[0])}

    target = {k1: np.concatenate([item[None,3:namecol] for item in list(g1)],
                                 axis = 0).astype('float') for k1,g1 in itertools.groupby(sorted(df.values,
                                                                                                      key = lambda x:x[0]),
                                                                                               lambda x: x[0])}
    names = {k1: np.concatenate([item[None,[namecol]] for item in list(g1)],
                                axis = 0)[:,0] for k1,g1 in itertools.groupby(sorted(df.values,
                                                                                     key = lambda x:x[0]),
                                                                              lambda x: x[0])}
    
    data = {k1: np.concatenate([item[None,namecol+1:] for item in list(g1)],
                                axis = 0).astype('float') for k1,g1 in itertools.groupby(sorted(df.values,
                                                                                                key = lambda x:x[0]),
                                                                                         lambda x: x[0])}
    
    return regions, target, names, data

        
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
        glist = from_regions(clrs,
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
            item['target'] = tdict[chrom][jdx,:]
            item['prom_x'] = prom_info[chrom][jdx,:]
            torch_item = ptg_from_npy(item)
            torch.save(torch_item, 
                       os.path.join(out_path,
                                    "data_{}_{}.pt".format(name_chr(chrom),
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
                 tracks=None,
                 names=None,
                 target=None,
                 track_transform=None,
                 transform=None,
                 pre_transform=None,
                 buffer = 25e4,
                 binsize = 1e4,
                 chunk_size = 500,
                 train_test_split = 0.3,
                 train = True,
                 random_state=42,
                 track_statistic_types=['mean'],
                 chromosomes = CHROMS
                ):
        #Estimate number of objects by rows of our target csv
        self.target = target
        if self.target is not None:
            self.num_objects = pd.read_table(os.path.join(root,
                                                      os.path.join('raw',
                                                                   self.target)
                                                     )
                                            ).values.shape[0]
        
        #Initialise class specific attributes
        self.contacts = contacts
        if self.contacts is not None:
            c = cooler.Cooler(os.path.join(root,
                                       os.path.join('raw/contacts',
                                                    self.contacts[0]
                                                   )
                                      )
                         )
            self.chr_lims = {str(row[0]):int(row[1]) for row in c.chroms()[:].values}
        
        if names is None:
            self.names = tracks
        else:
            self.names = names
            
        self.tracks = tracks    
        self.stats_types = track_statistic_types
        self.track_transform = track_transform
        self.buffer = buffer
        self.binsize = binsize
        self.numnodespergraph = int((2*self.buffer)/self.binsize)+1
        self.chunk_size = chunk_size
        self.chromosomes = chromosomes
        self.train_test_split = train_test_split
        self.train = True
        self.random_state = random_state
        
        #Initialise the class from the parent intialisation
        super(HiC_Dataset, self).__init__(root,
                                          transform,
                                          pre_transform)
                
    @property
    def raw_file_names(self):
        conts = [os.path.join('contacts',item) for item in self.contacts]
        bigwigs = [os.path.join('tracks',item) for item in self.tracks]
        return conts + tracks + [self.target]

    @property
    def processed_file_names(self):
        if self.train:
            data_paths = [os.path.join('train',
                                       os.path.split(item)[1]) for item in glob.glob(os.path.join(self.root,
                                                                                                  f'processed/train/data_*.pt')
                                                                                    )
                         ]
        else:
            data_paths = [os.path.join('test',
                                       os.path.split(item)[1]) for item in glob.glob(os.path.join(self.root,
                                                                                                  f'processed/test/data_*.pt')
                                                                                    )
                         ]
        return data_paths + ['cooler_track_data.csv']

    def process(self):
        full_contact_paths= [os.path.join(self.root,
                                          os.path.join('raw/contacts',
                                                       path
                                                      )
                                         ) for path in self.contacts]
        full_track_paths= [os.path.join(self.root,
                                         os.path.join('raw/tracks',
                                                      path
                                                     )
                                        ) for path in self.tracks]
        full_target_path = os.path.join(self.root,
                                        os.path.join('raw',
                                                     self.target
                                                    )
                                       )
        #EVALUATE BIGWIGS OVER COOLER BINS AND SAVE TO FILE
        track_out_file = os.path.join(self.processed_dir, "cooler_track_data.csv")
        print(f"Evaluating tracks over cooler bins and saving to file {track_out_file}")
        df = eval_tracks_over_cooler(full_contact_paths[0],
                                  paths = full_track_paths,
                                  names = self.names,
                                  stats_types = self.stats_types,
                                  allowed_chroms = self.chromosomes
                                 )
        
        if isinstance(self.track_transform,str):
            if self.track_transform == "Power":
                norm = PowerTransform_norm
            elif self.track_transform == "Standard":
                norm = Standard_norm
            elif self.track_transform == "Robust":
                norm = Robust_norm
        elif self.track_transform is not None:
            norm = lambda x: np.apply_along_axis(self.track_transform,
                                                 0,
                                                 x)
            
        if self.track_transform is not None:
            df = pd.DataFrame(data= norm(df.values.astype('float')),
                              columns = df.columns,
                              index= df.index)
        
        df.to_csv(track_out_file, 
                  sep = "\t")
        
        #EVALUATE BIGWIGS OVER THE TARGETS AND SAVE TO FILE
        print(f"Evaluating bigwigs over specific regions of interest and appending to target file {full_target_path}")
        df = pd.read_table(full_target_path)
        namecol = np.where(df.columns.values=='name')[0][0]+1
        df = df[df.columns.values[:namecol]]
        colnames, arr = eval_tracks_over_bed_df(df,
                                             full_track_paths,
                                             names = self.names,
                                             stats_types = self.stats_types)
        if self.track_transform is not None: 
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
                     track_out_file,
                     pdata,
                     self.chunk_size,
                     self.processed_dir)
        p = Pool()
        t_outs = p.imap(fn, (chrom for chrom in graph_regions))
        for t_out in t_outs:
            pass
            
        print("Renaming files and splitting data into train and test data")
        data_files = glob.glob(os.path.join(self.processed_dir,
                                           "*.pt")
                             )
        for idx, file in enumerate(data_files):
            dat = torch.load(file).x.shape[0]
            if dat != self.numnodespergraph:
                os.remove(file)
                
        data_files = glob.glob(os.path.join(self.processed_dir,
                                           "*.pt")
                             )
        train_files, test_files,_,_ = tts(data_files,
                                          np.ones(len(data_files)),
                                          test_size=self.train_test_split,
                                          random_state=self.random_state)
        for idx, file in enumerate(train_files):
            os.rename(file,
                      os.path.join(self.processed_dir,
                                   f"train/data_{idx}.pt")
                         )
        for idx,file in enumerate(test_files):
            os.rename(file,
                      os.path.join(self.processed_dir,
                                   f"test/data_{idx}.pt")
                     )

        
        
    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        if self.train:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'train/data_{idx}.pt'))
        else:
            data = torch.load(os.path.join(self.processed_dir,
                                           f'test/data_{idx}.pt'))
        return data