import numpy as np
from random import shuffle
from sklearn.metrics import (precision_score, 
                             recall_score,
                             balanced_accuracy_score,
                             roc_auc_score
                            )
from sklearn.utils.class_weight import compute_class_weight as ccw

from GrapHiC.models.recurrent_GATE_encoder import RGATE_Encoder
from GrapHiC.Dataset import HiC_Dataset

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW

import torch_geometric as tgm
from torch_geometric.data import DataLoader

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
CONVOLUTIONAL_DROPOUT = 0.01
MANUAL_SEED = 30


'''
LIGHTNING NET
'''
class LitGATENet(pl.LightningModule):
    def __init__(self,
                 module,
                 train_loader,
                 val_loader,
                 learning_rate,
                 numsteps,
                 criterion
                ):
        super().__init__()
        self.module = module
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.numsteps = numsteps
        self.criterion = criterion

    def train_dataloader(self):
        return self.train_loader
    
    def validation_dataloader(self):
        return self.test_loader

    def shared_step(self, batch):
        pred = self.module(batch).squeeze()
        loss = self.criterion(pred, 
                              batch.y.long())
        return loss, pred

    def customlog(self, 
                  name,
                  loss,
                  pred,
                  actual
                 ):
        
        cls = pred.argmax(dim=1)
        # identifying number of correct predections in a given batch
        correct=cls.eq(actual).sum().item()
        # identifying total number of labels in a given batch
        total=len(actual)
        
        cls = Tensor.cpu(cls)
        actual = Tensor.cpu(actual)
 
        precision = precision_score(cls.numpy(), 
                                    actual.numpy(),
                                    average = 'weighted')
        recall = recall_score(cls.numpy(), 
                              actual.numpy(),
                              average = 'weighted')
        
        adj_balanced_acc = balanced_accuracy_score(cls.numpy(),
                                               actual.numpy(),
                                               adjusted = True)
        balanced_acc = balanced_accuracy_score(cls.numpy(),
                                               actual.numpy(),
                                               adjusted = False)
        
        self.log(f"{name}_raw_acc",correct/total)
        self.log(f"{name}_adjusted_balanced_acc",adj_balanced_acc)
        self.log(f"{name}_unadjusted_balanced_acc",balanced_acc)
        self.log(f"{name}_weighted_average_precision",precision)
        self.log(f"{name}_weighted_average_recall", recall)
        self.log(f"{name}_loss",loss)

        
    def training_step(self, batch, batch_idx):
        loss, pred = self.shared_step(batch)
        self.customlog('train',
                       loss,
                       pred, 
                       batch.y)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred = self.shared_step(batch)
        self.customlog('val',
                       loss,
                       pred, 
                       batch.y)
        return loss

    def test_step(self, batch, batch_idx):
        loss, pred = self.shared_step(batch)
        self.customlog('test',
                       loss,
                       pred, 
                       batch.y)
        return loss

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), 
                                      lr=self.learning_rate)
        lr_scheduler = {'scheduler': OneCycleLR(
                                        optimizer,
                                        max_lr=10*self.learning_rate,
                                        total_steps = self.numsteps,
                                        anneal_strategy="cos",
                                        final_div_factor = 1e3,
                                    ),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}

        return [optimizer], [lr_scheduler]



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
    
        weights = [1,1,1]
        dset = newdset
        shuffle(dset)
        
        numdatapoints = len(dset)
        trainsize = int(numdatapoints*hparams.trainfraction)
        train_dset, val_dset = random_split(dset,
                                            [trainsize, numdatapoints-trainsize],
                                            generator=torch.Generator().manual_seed(MANUAL_SEED)
                                           )

    classes = ('up','down','nonsig')
    criterion = CrossEntropyLoss(weight = torch.tensor(weights).float())
    
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

    module = RGATE_Encoder(hidden_channels = hparams.hiddenchannels,
                         numchip = NUMCHIP,
                         numedge = NUMEDGE,
                         embedding_layers = hparams.embeddinglayers,
                         num_fc = hparams.fullyconnectedlayers,
                         fc_channels = hparams.fullyconnectedchannels,
                         num_graph_convs = hparams.graph_convolutions,
                         positional_encoding = hparams.positional_encoding,
                         pos_embedding_dropout = hparams.pdropout,
                         fc_dropout = hparams.fdropout,
                         conv_dropout = hparams.cdropout,
                           numnodespergraph = NUMNODESPERGRAPH
                        )
    Net = LitGATENet(module,
                     train_loader,
                     val_loader,
                     hparams.learning_rate,
                     hparams.numsteps,
                     criterion = criterion
                    )

    tb_logger = pl_loggers.TensorBoardLogger(hparams.logdir,
                                             name = hparams.experiment_name,
                                             version = hparams.version
                                             )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.epochs,
                         progress_bar_refresh_rate=1,
                         logger=tb_logger,
                         auto_lr_find=hparams.auto_lr_find,
                         resume_from_checkpoint=hparams.checkpoint,
                         callbacks=[lr_monitor],
                         stochastic_weight_avg=True
                         )
    
    if hparams.auto_lr_find:
        # Run learning rate finder
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
    parser.add_argument('--cdropout',
                        type = float,
                        default =CONVOLUTIONAL_DROPOUT)
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
                        default = 1000)
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
    args = parser.parse_args()

    main(args)
