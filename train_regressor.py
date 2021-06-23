import numpy as np
from src.Dataset import HiC_Dataset
from src.models.GrapHiC import GrapHiC_Encoder

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss
from torch.utils.data import random_split
from torch.optim.lr_scheduler import OneCycleLR

import torch_geometric as tgm
from torch_geometric.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

DATASET = "Data/test_dset_18features_custom_norm.pt"
NUMEPOCHS = 5000
NUMCLASSES = 3
BATCHSIZE = 200
LEARNING_RATE = 0.0001
POS_EMBEDDING_DROPOUT = 0.01
FULLY_CONNECTED_DROPOUT = 0.01
CONVOLUTIONAL_DROPOUT = 0.01
MANUAL_SEED = 30


'''
LIGHTNING NET
'''
class LitWEGATNet(pl.LightningModule):
    def __init__(self,
                 module,
                 train_loader,
                 val_loader,
                 learning_rate,
                 numsteps
                ):
        super().__init__()
        self.module = module
        self.learning_rate = learning_rate
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.numsteps = numsteps

    def train_dataloader(self):
        return self.train_loader
    def validation_dataloader(self):
        return self.test_loader


    def shared_step(self, batch):
        pred = self.module(batch).squeeze()
        loss = F.l1_loss(pred, batch.y.float())
        return loss, pred

    def customlog(self, name, loss, pred):
        self.log(f'{name}_loss', loss)
        self.log(f'{name}_maxabs_prediction',
                 torch.max(abs(pred)).item())
        self.log(f'{name}_mean_prediction',
                 torch.mean(pred).item())
        self.log(f'{name}_std_prediction',
                 torch.std(pred).item())

    def training_step(self, batch, batch_idx):
        loss, pred = self.shared_step(batch)
        self.customlog('train',loss, pred)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, pred = self.shared_step(batch)
        self.customlog('val',loss, pred)
        return loss

    def test_step(self, batch, batch_idx):
        loss, pred = self.shared_step(batch)
        self.customlog('test',loss, pred)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.learning_rate)
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': OneCycleLR(optimizer,
                                        max_lr = 10*self.learning_rate,
                                        total_steps = self.numsteps
                                       )
            }
        }

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
    print("Loading in memory datasets")
    dset = torch.load(hparams.dataset)

    numdatapoints = len(dset)
    trainsize = int(numdatapoints*hparams.trainfraction)
    train_dset, val_dset = random_split(dset,
                                        [trainsize, numdatapoints-trainsize],
                                        generator=torch.Generator().manual_seed(MANUAL_SEED)
                                       )

    print("Loaded in memory datasets")
    train_loader = DataLoader(train_dset,
                              batch_size=hparams.batchsize,
                              shuffle = True
                             )
    val_loader = DataLoader(val_dset,
                             batch_size=hparams.batchsize,
                             shuffle = True
                           )


    '''
    INITIALISING/TRAINING THE MODEL
    '''
    NUMCHIP = dset[0].x.shape[1]
    NUMEDGE = dset[0].edge_attr.shape[1]
    NUMNODESPERGRAPH = dset[0].x.shape[0]

    module = GrapHiC_Encoder(hidden_channels = hparams.hiddenchannels,
                         numchip = NUMCHIP,
                         numedge = NUMEDGE,
                         embedding_layers = hparams.embeddinglayers,
                         num_fc = hparams.fullyconnectedlayers,
                         fc_channels = hparams.fullyconnectedchannels,
                         num_graph_convs = hparams.graph_convolutions,
                         positional_encoding = hparams.positional_encoding,
                         pos_embedding_dropout = hparams.pdropout,
                         fc_dropout = hparams.fdropout,
                         conv_dropout = hparams.cdropout
                        )
    Net = LitGATENet(module,
                      train_loader,
                      val_loader,
                      hparams.learning_rate,
                      hparams.numsteps,
                      criterion = criterion)

    tb_logger = pl_loggers.TensorBoardLogger(hparams.logdir,
                                             name = hparams.experiment_name,
                                             version = hparams.version
                                             )
    trainer = pl.Trainer(gpus=hparams.gpus,
                         max_epochs=hparams.epochs,
                         progress_bar_refresh_rate=1,
                         logger=tb_logger,
                         auto_lr_find=hparams.auto_lr_find,
                         resume_from_checkpoint=hparams.checkpoint,
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
                        default = 3000)
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
    parser.add_argument('-gc',
                        '--graph_convolutions',
                        type = int,
                        default = 6)
    args = parser.parse_args()

    main(args)
