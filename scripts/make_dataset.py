#!/home/dh486/gnn-env/bin/python
from utils.datasets import pop_HiC_Dataset

dset = pop_HiC_Dataset("/home/dh486/rds/hpc-work/GNN_Work/data/",
                       condition = 'SLX-7671_haploid',
                       target = {'totalRNA.npz':None},
                       targetIDs = False,
                       target_mask = True,
                       binSize = 50)

