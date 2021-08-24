import numpy as np
from random import shuffle
from sklearn.utils.class_weight import compute_class_weight as ccw

from GrapHiC.models.GATE_modules import GATE_promoter_module
from GrapHiC.models.lightning_nets import LitClassifierNet
from GrapHiC.Dataset import HiC_Dataset

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split

import torch_geometric as tgm
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import LearningRateMonitor

DATASET = "Data/dset_15difffeatures_500kb_wayneRNAseq.pt"
NUMEPOCHS = 5000
NUMCLASSES = 3
BATCHSIZE = 100
LEARNING_RATE = 0.0001
POS_EMBEDDING_DROPOUT = 0.01
FULLY_CONNECTED_DROPOUT = 0.01
EDGE_DROPOUT = 0.1
MANUAL_SEED = 30


'''
MAIN FUNCTION
'''
def main(hparams):
    if hparams.hiddenchannels%2 !=0 and hparams.positional_encoding:
        print("positional encoding requires an even number of hidden channels, adding one to hidden channels")
        hparams.hiddenchannels += 1
    '''
    CONSTRUCTING THE DATALOADERS
    '''
    print("Loading datasets")
    if hparams.inmemory != 1:
        train_dset = HiC_Dataset("Data")
        val_dset = HiC_Dataset("Data",train= False)
    
        print("Calculating class weights")
        vals = np.array([d.y[0,1].item() for d in train_dset])
        nums = [np.sum(np.array(vals)==idx) for idx in [0,1,2]]
        weights = ccw('balanced', 
                  np.array([0.0,1.0,2.0]),
                  vals)
    else:
        dset = torch.load(hparams.dataset)
        
        print("Calculating class weights")
        #vals = np.array([d.y[0,1].item() for d in dset])
        #nums = [np.sum(np.array(vals)==idx) for idx in [0,1,2]]
        #weights = ccw('balanced', 
        #          np.array([0.0,1.0,2.0]),
        #          vals)
        
        vals = np.array([d.y.item() for d in dset])
        idxs = {'up': np.argsort(vals)[-5000:],
                'down': np.argsort(vals)[:5000],
                'nonsig': np.argsort(abs(vals))[:5000]
               }
        newdset = []
        for cls in ['up','down','nonsig']:
            for idx in idxs[cls]:
                d_add = dset[idx]
                if cls == 'up':
                    d_add.y = 0.0
                elif cls == 'down':
                    d_add.y = 1.0
                else:
                    d_add.y = 2.0
                newdset.append(d_add)
    
        dset = newdset
        shuffle(dset)
        
        numdatapoints = len(dset)
        trainsize = int(numdatapoints*hparams.trainfraction)
        train_dset, val_dset = random_split(dset,
                                            [trainsize, numdatapoints-trainsize],
                                            generator=torch.Generator().manual_seed(MANUAL_SEED)
                                           )

    classes = ('up','down','nonsig')
    criterion = CrossEntropyLoss()
    
    print("Loaded in memory datasets")
    train_loader = DataLoader(train_dset,
                              batch_size=hparams.batchsize,
                              shuffle = True,
                              drop_last=True
                             )
    val_loader = DataLoader(val_dset,
                             batch_size=hparams.batchsize,
                             shuffle = True,
                            drop_last=True
                           )
    '''
    INITIALISING/TRAINING THE MODEL
    '''
    NUMCHIP = dset[0].x.shape[1]
    NUMEDGE = dset[0].edge_attr.shape[1]
    NUMNODESPERGRAPH = dset[0].x.shape[0]
    
    if hparams.dropout is not None:
        hparams.pdropout = hparams.dropout
        hparams.fdropout = hparams.dropout
    
    if hparams.recurrent == 1:
        hparams.recurrent = True
    else:
        hparams.recurrent = False
        
    if hparams.principal_neighbourhood_aggregation == 1:
        hparams.principal_neighbourhood_aggregation = True
        # Compute in-degree histogram over training data.
        deg = torch.zeros(NUMNODESPERGRAPH+1, dtype=torch.long)
        for data in train_dset:
            d = degree(data.edge_index[1], 
                       num_nodes=data.num_nodes, 
                       dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
    else:
        hparams.principal_neighbourhood_aggregation = False
        deg = None
        
    module = GATE_promoter_module(hidden_channels = hparams.hiddenchannels,
                                  inchannels = NUMCHIP,
                                  edgechannels = NUMEDGE,
                                  embedding_layers = hparams.embeddinglayers,
                                  num_fc = hparams.fullyconnectedlayers,
                                  fc_channels = hparams.fullyconnectedchannels,
                                  num_graph_convs = hparams.graph_convolutions,
                                  positional_encoding = hparams.positional_encoding,
                                  pos_embedding_dropout = hparams.pdropout,
                                  fc_dropout = hparams.fdropout,
                                  edge_dropout = hparams.edropout,
                                  recurrent = hparams.recurrent,
                                  numnodespergraph = NUMNODESPERGRAPH,
                                  principal_neighbourhood_aggregation = hparams.principal_neighbourhood_aggregation,
                                  deg = deg,
                                  aggr = hparams.aggregation,
                                  heads = hparams.heads
                                 )
    Net = LitClassifierNet(module,
                     train_loader,
                     val_loader,
                     criterion = criterion,
                     inputhparams = hparams
                    )

    tb_logger = pl_loggers.TensorBoardLogger(hparams.logdir,
                                             name = hparams.experiment_name,
                                             version = hparams.version
                                             )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    if hparams.plot_lr and hparams.auto_lr_find:
        trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.epochs,
                         progress_bar_refresh_rate=1,
                         logger=tb_logger,
                         auto_lr_find=False,
                         resume_from_checkpoint=hparams.checkpoint,
                         callbacks=[lr_monitor],
                         stochastic_weight_avg=True
                         )
    
        lr_finder = trainer.tuner.lr_find(Net)

        # Results can be found in
        lr_finder.results

        # Plot with
        if hparams.plot_lr:
            fig = lr_finder.plot(suggest=True)
            fig.savefig("learning_rate_suggestion.png", format = 'png')

        # Pick point based on plot, or get suggestion
        new_lr = lr_finder.suggestion()
        # update hparams of the model
        Net.hparams.lr = new_lr
    elif hparams.auto_lr_find:
        trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.epochs,
                         progress_bar_refresh_rate=1,
                         logger=tb_logger,
                         auto_lr_find=True,
                         resume_from_checkpoint=hparams.checkpoint,
                         callbacks=[lr_monitor],
                         stochastic_weight_avg=True
                         )
        
        trainer.tune(Net)
    else:
        trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.epochs,
                         progress_bar_refresh_rate=1,
                         logger=tb_logger,
                         auto_lr_find=False,
                         resume_from_checkpoint=hparams.checkpoint,
                         callbacks=[lr_monitor],
                         stochastic_weight_avg=True
                         )
        Net.hparams.lr = hparams.learning_rate
        
    trainer.fit(Net, train_loader, val_loader)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-g',
                        '--gpus',
                        type = int,
                        default=1)
    parser.add_argument('-hidden',
                        '--hiddenchannels',
                        type = int,
                        default=10)
    parser.add_argument('-heads',
                        '--heads',
                        type = int,
                        default=4)
    parser.add_argument('-em',
                        '--embeddinglayers',
                        type = int,
                        default=10)
    parser.add_argument('-e',
                        '--epochs',
                        type = int,
                        default=NUMEPOCHS)
    parser.add_argument('-l',
                        '--logdir',
                        type = str,
                        default='runs/')
    parser.add_argument('-b',
                        '--batchsize',
                        type = int,
                        default=BATCHSIZE)
    parser.add_argument('-d',
                        '--dataset',
                        type = str,
                        default=DATASET)
    parser.add_argument('-t',
                        '--trainfraction',
                        type = float,
                        default=0.7)
    parser.add_argument('--learning_rate',
                        type = float,
                        default=LEARNING_RATE)
    parser.add_argument('--pdropout',
                        type = float,
                        default=POS_EMBEDDING_DROPOUT)
    parser.add_argument('--fdropout',
                        type = float,
                        default=FULLY_CONNECTED_DROPOUT)
    parser.add_argument('--edropout',
                        type = float,
                        default =EDGE_DROPOUT)
    parser.add_argument('-c',
                        '--checkpoint',
                        default=None)
    parser.add_argument('-alr',
                        '--auto_lr_find',
                        type = bool,
                        default=False)
    parser.add_argument('-plr',
                        '--plot_lr',
                        type = bool,
                        default=False)
    parser.add_argument('-p',
                        '--positional_encoding',
                        type = bool,
                        default=True)
    parser.add_argument('-pna',
                        '--principal_neighbourhood_aggregation',
                        type = int,
                        default=0)
    parser.add_argument('-m',
                        '--manual_seed',
                        type = int,
                        default=MANUAL_SEED)
    parser.add_argument('-v',
                        '--version',
                        type = int,
                        default = 0)
    parser.add_argument('-en',
                        '--experiment_name',
                        type = str,
                        default = 'default')
    parser.add_argument('-n',
                        '--numsteps',
                        type = int,
                        default = int(1e6))
    parser.add_argument('-gc',
                        '--graph_convolutions',
                        type = int,
                        default = 6)
    parser.add_argument('-fcc',
                        '--fullyconnectedchannels',
                        type = int,
                        default = 10)
    parser.add_argument('-fcl',
                        '--fullyconnectedlayers',
                        type = int,
                        default = 10)
    parser.add_argument('-im',
                        '--inmemory',
                        type = int,
                        default = 1)
    parser.add_argument('-dr',
                        '--dropout',
                        type = float,
                        default = None)
    parser.add_argument('-r',
                        '--recurrent',
                        type = int,
                        default = 1)
    parser.add_argument('-a',
                        '--aggregation',
                        type = str,
                        default = 'add')
    args = parser.parse_args()

    main(args)
