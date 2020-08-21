import torch
from torch_geometric.data import InMemoryDataset, Data
from sklearn.model_selection import train_test_split
import numpy as np
import os.path as osp   

from utils.processing import make_ptg_pophic_edge_index as make_index
        
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class pop_HiC_Dataset(InMemoryDataset):
    ''' 
    Class to representing/constructing a pytorch geometric dataset from a contact file alongside node features such as
    ChIP-seq, GC content, DNA methylation, raw sequence information etc.
    
    Arguments:
    
    - root: Root of the filesystem in which the processed data files and unprocessed raw files for construction are held. 
            The file system is assumed to be of the form:
            
            {root}/data/
            {root}/data/{condition}/
            {root}/data/{condition}/processed/
            {root}/data/{condition}/raw/
            {root}/data/{condition}/raw/single_cells/
            
            (for info about {condition}, see the condition argument for __init__).
            
    - transform: Dynamically transforms the data object before accessing (best used for data augmentation).
    - pre_transform: Applies the transformation before saving the data objects to disk (best used for heavy
                     precomputation which needs to be only done once).
    - condition: The experimental condition under which the contact data was gathered. For example, in the Laue lab case
                 we have SLX-7671_haploid corresponding to the population Hi-C experiment on haploid mESCs. This condition
                 name will also be used to store the processed dataset object as. Note that the constructor expects the
                 base Hi-C contact file to be stored as:
                               {root}/data/{condtion}/raw/{condition}_{binSize}kb.npz
                 (For information about {binSize} see the binSize argument in the constructor).
    - binSize: The genomic length of each chromatin bin (in kilobases). Defaults to 50kb.
    - chrlims: Dictionary detailing the start and end basepairs of each chromosome in the contact dictionary. NOTE: the
                chromosome limits are inclusive i.e. for each CHR_A we should have chromo_limits[CHR_A] = (start_A, end_A)
                where all basepairs b on this chromsome satisfy:
                                 start_A <= b <= end_A
                
                This input shouldnt need to be used at the moment but may be expanded on in further edits. 
    - shape: Shape of the resulting contact/adjacency matrix. (This shouldnt need to be input).
     
    Returns:
     
    If the data has already be pre-processed then initialising this dataset should return a dataset object with the node
    features specified in the document:
                {root}/data/{condition}/processed/feature_indexes.npz
     
    The first 20 features should be common to all datasets as the chromosome-basepair position encoded as a length-20
    one-hot chromosome encoding.
     
    If the data has not been pre-processed then the constructor will search for the relevant files in:
                {root}/data/{condition}/raw/
    and then construct the correct dataset object in:
                {root}/data/{condition}/processed/
     
    NOTE: If you have previously constructed a dataset object but would like to include more features in some new dataset
           then you must delete the contents of:
                {root}/data/{condition}/processed/
           and reconstruct the dataset from scratch with the correct features to be included. 
            
    
    '''
    def __init__(self,
                 root,
                 transform=None,
                 pre_transform=None,
                 condition = 'SLX-7671_haploid',
                 binSize = '50',
                 chrlims = None,
                 shape = None,
                 target_mask = False
                ):
        
        self.binSize = binSize
        self.chrlims = chrlims
        self.shape = shape
        
        self.condition = condition
        
        super(pop_HiC_Dataset, self).__init__(root + "{}".format(self.condition), transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        print("Looking for files...")
        return ['{}_{}kb.npz'.format(self.condition, self.binSize)]

    @property
    def processed_file_names(self):
        return ['{}_{}kb_data.pt'.format(self.condition, int(self.binSize))]

    def download(self):
        pass

    def process(self): 
        #Retrieve contact information from contact npz file
        self.shape, self.binSize, self.chrlims, edge_index, edge_attr = make_index(osp.join(self.raw_dir, '{}_{}kb.npz'.format(self.condition, self.binSize)))
        
        #Set up basic feature matrix - one-hot chromosome encoding and bp bin positions
        fmat = torch.empty(self.shape, 0, dtype=torch.float)

        #create dummy target
        tvals = torch.ones(self.shape, dtype = torch.float)
        
        #create our dataset object
        data = Data(x = fmat, edge_index = edge_index, edge_attr = edge_attr, y = tvals)

        #dummy target mask
        mymask = np.arange(tvals.shape)
            
        train, other = train_test_split(mymask, test_size = 0.5, shuffle = True)
        test, val = train_test_split(other, test_size = 0.5, shuffle = True)
        data.train_mask = torch.tensor(train, dtype = torch.long)
        data.test_mask = torch.tensor(test, dtype = torch.long)
        data.val_mask = torch.tensor(val, dtype = torch.long)
        
        
        data_list = [data]

        #Reset binSize back to kb units
        self.binSize /= 1000
        
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        
        return data
