import numpy as np
from random import shuffle

from GrapHiC.models.GATE_modules import GATE_variational_encoder, GATE_decoder
from GrapHiC.models.autoencoder import ARGVA
from GrapHiC.Dataset import HiC_Dataset

import torch
from torch import Tensor
from torch.nn import CrossEntropyLoss, L1Loss
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
BATCHSIZE = 100
LEARNING_RATE = 0.0001
POS_EMBEDDING_DROPOUT = 0.01
FULLY_CONNECTED_DROPOUT = 0.01
CONVOLUTIONAL_DROPOUT = 0.01
MANUAL_SEED = 30

class Discriminator(torch.nn.Module):
    def __init__(self, 
                 latentchannels = 10,
                 hiddenchannels = 10,
                 numlayers = 3,
                 dropout = 0.01
                ):
        super(Discriminator, self).__init__()
        #fully connected channels
        fc_channels = [hiddenchannels]*numlayers
        fc_channels = [latentchannels]+fc_channels
        fc_channels = fc_channels + [1]
        lin = []
        for idx in torch.arange(numlayers):
            lin.append(BatchNorm1d(fc_channels[idx]))
            lin.append(torch.nn.ReLU())
            lin.append(torch.nn.Dropout(p=dropout))
            lin.append(Linear(fc_channels[idx],fc_channels[idx+1]))
        self.lin = Sequential(*lin)

    def forward(self, z):
        return self.lin(z)

'''
LIGHTNING NET
'''
class LitARGVA(pl.LightningModule):
    def __init__(self,
                 encoder,
                 node_decoder,
                 discriminator,
                 negsampling,
                 numnodespergraph,
                 learning_rate,
                 numlrsteps,
                 criterion
                ):
        super().__init__()
        self.module = ARGVA(encoder, 
                            discriminator, 
                            node_decoder)
        self.numsteps = numlrsteps
        self.criterion = criterion
        self.lr = learning_rate
        self.negsampling = negsampling
        self.numnodespergraph = numnodespergraph
        
        # cache for generated latents
        self.generated_latents = None

    def shared_step(self, batch, name, optimizer_idx):
        # train generator
        if optimizer_idx == 0:
            #store original graph descriptors
            x = b.x
            ea = b.edge_attr
            ei = b.edge_index
            #variational encoding
            self.generated_latents = self.module.encode(b)
            #restore input edge features and node features
            b.edge_index = ei.long()
            b.x = x
        
            recon_loss = self.module.VGAE.recon_loss(self.generated_latents, 
                                                 b,
                                                 negsampling = self.negsampling, 
                                                 nodespergraph = self.numnodespergraph)
            kl_loss = (1/b.num_nodes)*self.module.kl_loss()
            reg_loss = self.module.reg_loss(self.generated_latents)
            
            self.log(f"{name}_recon_loss",recon_loss)
            self.log(f"{name}_kl_loss",kl_loss)
            self.log(f"{name}_reg_loss",reg_loss)
            
            vgae_loss = recon_loss + kl_loss + reg_loss
            
            self.log(f"{name}_VGAE_loss",vgae_loss)
            
            tqdm_dict = {f'{name}_VGAE_loss': vgae_loss}
            output = OrderedDict({
                'loss': vgae_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output
        
        # train discriminator
        if optimizer_idx == 1:
            disc_loss = self.module.discriminator_loss(self.generated_latents)
            
            tqdm_dict = {f'{name}_d_loss': disc_loss}
            output = OrderedDict({
                'loss': disc_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
            return output

        
    def training_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        return output

    def validation_step(self, batch, batch_idx):
        output = self.shared_step(batch)
        return output

    def test_step(self, batch, batch_idx):
        return 0

    def configure_optimizers(self):
        VGAE_optimizer = AdamW(self.module.VGAE.parameters(), 
                          lr=self.lr)
        VGAE_lr_scheduler = {'scheduler': OneCycleLR(
                                        VGAE_optimizer,
                                        max_lr=10*self.lr,
                                        total_steps = self.numsteps,
                                        anneal_strategy="cos",
                                        final_div_factor = 10,
                                    ),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}
        
        disc_optimizer = AdamW(self.module.discriminator.parameters(), 
                          lr=self.lr)
        disc_lr_scheduler = {'scheduler': OneCycleLR(
                                        disc_optimizer,
                                        max_lr=10*self.lr,
                                        total_steps = self.numsteps,
                                        anneal_strategy="cos",
                                        final_div_factor = 10,
                                    ),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}

        return [
            {'optimizer': VGAE_optimizer, 'frequency': 10,'lr_scheduler':VGAE_lr_scheduler},
            {'optimizer': disc_optimizer, 'frequency': 10,'lr_scheduler':disc_lr_scheduler}
        ]


class MyDataModule(pl.LightningDataModule):
    def __init__(self,
                 dset,
                 trainfraction,
                 batch_size: int = 32):
        super().__init__()
        self.dset = torch.load(dset)
        numdatapoints = len(self.dset)
        trainsize = int(numdatapoints*trainfraction)
        self.train_dset, self.val_dset = random_split(self.dset,
                                            [trainsize, numdatapoints-trainsize],
                                            generator=torch.Generator().manual_seed(MANUAL_SEED)
                                           )
        self.batch_size = batch_size
        self.inchannels = dset[0].x.shape[1]
        self.edgechannels = dset[0].edge_attr.shape[1]
        self.numnodespergraph = dset[0].x.shape[0]

    def train_dataloader(self):
        return DataLoader(self.train_dset,
                          batch_size=self.batchsize,
                          shuffle = True,
                          drop_last=True
                         )

    def val_dataloader(self):
        return DataLoader(self.val_dset,
                          batch_size=self.batchsize,
                          shuffle = True,
                          drop_last=True
                         )
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
    datamod = MyDataModule(hparams.dataset,
                           hparams.trainfraction,
                           hparams.batchsize)

    '''
    INITIALISING/TRAINING THE MODEL
    '''
    if hparams.dropout is not None:
        hparams.pdropout = hparams.dropout
        hparams.fdropout = hparams.dropout
    
    if hparams.recurrent == 1:
        hparams.recurrent = True
    else:
        hparams.recurrent = False
    
    if hparams.negsampling == 1:
        hparams.negsampling = True
    else:
        hparams.negsampling = False
        
    encoder = GATE_variational_encoder(hidden_channels = hparams.graphconvolution_hiddenchannels,
                                       inchannels = datamod.inchannels,
                                       edgechannels = datamod.edgechannels,
                                       outchannels = hparams.latentchannels,
                                       embedding_layers = hparams.embeddinglayers,
                                       num_fc = hparams.fullyconnectedlayers,
                                       fc_channels = hparams.fullyconnectedchannels,
                                       num_graph_convs = hparams.graph_convolutions,
                                       positional_encoding = hparams.positional_encoding,
                                       pos_embedding_dropout = hparams.pdropout,
                                       fc_dropout = hparams.fdropout,
                                       conv_dropout = hparams.cdropout,
                                       recurrent = hparams.recurrent,
                                       numnodespergraph = datamod.numnodespergraph)
    
    node_decoder = GATE_decoder(hidden_channels = hparams.graphconvolution_hiddenchannels,
                                inchannels = hparams.latentchannels,
                                edgechannels = datamod.edgechannels,
                                outchannels = hparams.latentchannels,
                                embedding_layers = hparams.embeddinglayers,
                                num_fc = hparams.fullyconnectedlayers,
                                fc_channels = hparams.fullyconnectedchannels,
                                num_graph_convs = hparams.graph_convolutions,
                                positional_encoding = hparams.positional_encoding,
                                pos_embedding_dropout = hparams.pdropout,
                                fc_dropout = hparams.fdropout,
                                conv_dropout = hparams.cdropout,
                                recurrent = hparams.recurrent,
                                numnodespergraph = datamod.numnodespergraph)
            
    discriminator = Discriminator(latentchannels = hparams.latentchannels,
                                  hiddenchannels = hparams.discriminator_hiddenchannels,
                                  numlayers = hparams.discriminator_layers,
                                  dropout = hparams.fdropout)
            
    Net = LitGATENet(encoder,
                     node_decoder,
                     discriminator,
                     hparams.negsampling,
                     datamod.numnodespergraph,
                     hparams.learning_rate,
                     hparams.numsteps,
                     criterion = criterion
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
        
    trainer.fit(Net, datamod)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('-g',
                        '--gpus',
                        type = int,
                        default=1)
    parser.add_argument('-gchidden',
                        '--graphconvolution_hiddenchannels',
                        type = int,
                        default=20)
    parser.add_argument('-dhidden',
                        '--discriminator_hiddenchannels',
                        type = int,
                        default=10)
    parser.add_argument('-dl',
                        '--discriminator_layers',
                        type = int,
                        default=3)
    parser.add_argument('-la',
                        '--latentchannels',
                        type = int,
                        default=10)
    parser.add_argument('-em',
                        '--embeddinglayers',
                        type = int,
                        default=3)
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
                        default = int(1e6))
    parser.add_argument('-gc',
                        '--graph_convolutions',
                        type = int,
                        default = 3)
    parser.add_argument('-fcc',
                        '--fullyconnectedchannels',
                        type = int,
                        default = 10)
    parser.add_argument('-fcl',
                        '--fullyconnectedlayers',
                        type = int,
                        default = 3)
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
    parser.add_argument('-neg',
                        '--negsampling',
                        type = int,
                        default = 1)
    args = parser.parse_args()

    main(args)
