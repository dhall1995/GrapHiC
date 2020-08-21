#!/home/dh486/gnn-env/bin/python
from utils.Dataset import pop_HiC_Dataset

dset = pop_HiC_Dataset("/home/dh486/rds/hpc-work/GNN_Work/data/",
                       condition = 'SLX-7671_haploid',
                       binSize = 50)

