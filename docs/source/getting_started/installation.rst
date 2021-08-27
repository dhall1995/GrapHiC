Installation
============
GraphiC depends on a number of other libraries for constructing Hi-C graphs as well as for using Hi-C specific message-passing schemes. These should be installed in advance.

.. note::
    We recommend installing GrapHiC in a virtual environment.
    ..

.. note::
    Some of these packages have more involved setup depending on your requirements (i.e. CUDA). Please refer to the original packages for more detailed information
    
Creating Conda Environment
-----------------------------

.. code-block:: bash

    conda create -n graphic python=3.7

Installing PyTorch
------------------

.. code-block:: bash

    # Install PyTorch: MacOS
    $ conda install pytorch torchvision -c pytorch                      # Only CPU Build
    
    # Install PyTorch: Linux
    $ conda install pytorch torchvision cpuonly -c pytorch              # For CPU Build
    $ conda install pytorch torchvision cudatoolkit=9.2 -c pytorch      # For CUDA 9.2 Build
    $ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch     # For CUDA 10.1 Build
    $ conda install pytorch torchvision cudatoolkit=10.2 -c pytorch     # For CUDA 10.2 Build

Installing Pytorch Geometric
------------------------------
.. code-block:: bash

    $ pip install torch-scatter==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
    $ pip install torch-sparse==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
    $ pip install torch-cluster==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
    $ pip install torch-spline-conv==latest+${CUDA} -f https://pytorch-geometric.com/whl/torch-${TORCH}.html
    $ pip install torch-geometric

Install all needed packages with ${CUDA} replaced by either cpu, cu92, cu100 or cu101 depending on your PyTorch installation. 

.. note::
    Follow the instructions in the Torch-Geometric Docs (https://pytorch-geometric.readthedocs.io/en/latest/notes/installation.html) to install the versions appropriate to your CUDA version.

Install Cython & git-lfs
------------------------

.. code-block:: bash

     $pip install Cython
     $conda install git-lfs

Clone the git repo
------------------

.. code-block:: bash

    git clone https://github.com/dhall1995/GrapHiC
    cd GrapHiC
    pip install -e .


Setup Notebook graphic kernel (optional)
----------------------------------------
Optionally, to run the tutorial notebooks, run the following from within the conda environment:
.. code-block:: bash

    conda install -c anaconda ipykernel
    python -m ipykernel install --user --name=graphic
    
Then when starting a jupyter notebook choose the graphic kernel




