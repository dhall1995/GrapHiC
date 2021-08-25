# GrapHiC (Graph-based Hi-C)
[![Project Status: Active – The project has reached a stable, usable state and is being actively developed.](https://www.repostatus.org/badges/latest/active.svg)](https://www.repostatus.org/#active)

![banner](https://github.com/dhall1995/GrapHiC-ML/blob/master/workflow.png)
This package provides functionality for producing bespoke graph datasets using multi-omics data and Hi-C. We provide compatibility with the standard [bigwig](https://genome.ucsc.edu/goldenpath/help/bigWig.html) and [bed](https://genome.ucsc.edu/FAQ/FAQformat.html#format1) UCSC data formats and integrate with the popular [Cooler](https://cooler.readthedocs.io/en/latest/) format for Hi-C data.

## Example usage
### Creating a Hi-C graph from 3 disconnected regions of chromatin
```python
from GrapHiC.Graph_creation import from_regions

# Specify cooler paths
coolerpath ={'WT':'/disk2/dh486/cooler_files/WT/manually_downsampled/KR/KR_downsampled_WT_merged_10000.cool',
             'KO':'/disk2/dh486/cooler_files/KO/KR/KR_KO_merged_10000.cool'
            }

x = from_regions([coolerpath['WT'],coolerpath['KO']],         #paths to cooler files for edge featurisation
                 {'chr2':np.array([[8651256,10658971],        #Specifying region boundaries in chromosome 2, region 1
                                   [10678978,11658978],       #region 2
                                   [12678978,13658978],       #region 3
                                   [14678978,15658978],       #region 4
                                   [16678978,17658978]]),     #region 5
                  'chr3':np.array([[8651256,10658971],        #chromosome 3 region 1
                                   [10678978,11658978]])      #region 2
                 },
                 names = {'chr2': ['Ppp2r2c','reg2'],         #specifying names for the first two regions on chr2
                          'chr3': ['Ppp3r3c','reg4']          #specifying names for the first two regions on chr3
                         },
                 balance = True,                              #specifying whether to balance Hi-C matrices before extracting contact counts
                 join = False,                                #whether to join resulting dictionary of graphs into one graph object
                 backbone = True,                             #whether to force contacts to exist along the chromatin backbone
                 record_backbone_interactions = True,         #whether to explicitely record backbone interactions as an additional edge feature
                 record_cistrans_interactions = False,        #whether to calculate contacts across the interface of two regions (useful if joining region graphs)
                 record_node_chromosome_as_onehot = False     #whether to record a basic node feature as a one-hot encoding of the node chromosome
                )
```

### Creating a DataTrack and evaluating over cooler bins
```python
from GrapHiC.Datatrack_creation import evaluate_tracks_over_cooler_bins

#specify paths
contact_path = '/disk2/dh486/cooler_files/WT/manually_downsampled/KR/KR_downsampled_WT_merged_10000.cool'
bed_peaks = "tutorial_data/GSEXXXXXHendrich20161026_Nanog_ESC_peaks.bed"
bigwig = "tutorial_data/GSE71932_Nanog_mESCs_treat_pileup_filter.bw"

dataframe = evaluate_tracks_over_cooler_bins(contact_path,                                #specify cooler path
                                             paths = [nanog_peaks,nanog_bigwig],          #specify bed or bigwig paths
                                             names = ['nanog_bigwig','nanog_peaks'],      #specify track names
                                             stats_types = ['mean','max','std'],          #specify bin statistics to collect
                                             allowed_chroms = ['chr1','chr2','chrX'],     #specify chromosomes
                                             value_col = 4)                               #speicfy the target column of any BED data tracks 
dataframe.to_csv("example.csv",
                 sep="\t", 
                 index = False)
```

### Adding binned data to a list of graphs
```python
from GrapHiC.Graph_creation import add_binned_data_to_graphlist

add_binned_data_to_graphlist(x['chr2'],
                            'example.csv')
```

# Installation
1. Create env:

    ```bash
    conda create --name graphic python=3.7
    conda activate graphic
    ```
2. Install [PyTorch](https://pytorch.org):
    
    N.B. Make sure to install appropriate version for your CUDA version

    ```bash
    # Install PyTorch: MacOS
    $ conda install pytorch torchvision -c pytorch                      # Only CPU Build
    
    # Install PyTorch: Linux
    $ conda install pytorch torchvision cpuonly -c pytorch              # For CPU Build
    $ conda install pytorch torchvision cudatoolkit=9.2 -c pytorch      # For CUDA 9.2 Build
    $ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch     # For CUDA 10.1 Build
    $ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch     # For CUDA 10.2 Build
    ```


3. Install [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html):



    ```bash
    $ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
    $ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
    $ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
    $ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
    $ pip install torch-geometric
    ```
   Where `${CUDA}` and `${TORCH}` should be replaced by your specific CUDA version (`cpu`, `cu92`, `cu101`, `cu102`) and PyTorch version (`1.4.0`, `1.5.0`, `1.6.0`), respectively 
   
   N.B. Follow the [instructions in the Torch-Geometric Docs](https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install the versions appropriate to your CUDA version.

4. install cython
   ```
   pip install Cython
```   
4. Clone the git repo
   ```bash
   git clone https://github.com/dhall1995/GrapHiC-ML
   ```

5. install
   ```bash
   pip install -e .
  ```
  
For enquiries please email: dh486@cam.ac.uk
