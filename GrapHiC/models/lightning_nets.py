from collections import OrderedDict
import pytorch_lightning as pl
from sklearn.metrics import (precision_score, 
                             recall_score,
                             balanced_accuracy_score,
                             roc_auc_score
                            )

from .autoencoder import ARGVA

import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from torch import Tensor
from argparse import Namespace

class LitClassifierNet(pl.LightningModule):
    def __init__(self,
                 module,
                 train_loader,
                 val_loader,
                 criterion,
                 inputhparams
                ):
        super().__init__()
        '''
        self.save_hyperparameters(Namespace(**{'hiddenchannels': inputhparams.hiddenchannels,
                                               'embeddinglayers':inputhparams.embeddinglayers,
                                               'batchsize':inputhparams.batchsize,
                                               'dataset':inputhparams.dataset,
                                               'learning_rate': inputhparams.learning_rate,
                                               'positional_encoding': inputhparams.positional_encoding,
                                               'graph_convolutions': inputhparams.graph_convolutions,
                                               'fullyconnectedchannels': inputhparams.fullyconnectedchannels,
                                               'fullyconnectedlayers': inputhparams.fullyconnectedlayers,
                                               'dropout': inputhparams.dropout,
                                               'recurrent': inputhparams.recurrent,
                                               'numsteps':inputhparams.numsteps
                                              }
                                           )
                                 )
        '''
        self.save_hyperparameters(inputhparams)
        self.module = module
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.numsteps = inputhparams.numsteps
        self.criterion = criterion
        self.lr = inputhparams.learning_rate

    def train_dataloader(self):
        return self.train_loader
    
    def val_dataloader(self):
        return self.val_loader

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
                          lr=self.lr)
        lr_scheduler = {'scheduler': OneCycleLR(
                                        optimizer,
                                        max_lr=5*self.lr,
                                        total_steps = self.numsteps,
                                        anneal_strategy="cos",
                                        final_div_factor = 100,
                                    ),
                        'name': 'learning_rate',
                        'interval':'step',
                        'frequency': 1}

        return [optimizer], [lr_scheduler]

    
class LitARGVA(pl.LightningModule):
    def __init__(self,
                 encoder,
                 node_decoder,
                 discriminator,
                 numnodespergraph,
                 inputhparams
                ):
        super().__init__()
        '''
        self.save_hyperparameters(Namespace(**{'graphconvolution_hiddenchannels': inputhparams.graphconvolution_hiddenchannels,
                                               'discriminator_hiddenchannels':inputhparams.discriminator_hiddenchannels,
                                               'discriminator_layers': inputhparams.discriminator_layers,
                                               'latentchannels': inputhparams.latentchannels,
                                               'embeddinglayers':inputhparams.embeddinglayers,
                                               'batchsize':inputhparams.batchsize,
                                               'dataset':inputhparams.dataset,
                                               'learning_rate': inputhparams.learning_rate,
                                               'positional_encoding': inputhparams.positional_encoding,
                                               'graph_convolutions': inputhparams.graph_convolutions,
                                               'fullyconnectedchannels': inputhparams.fullyconnectedchannels,
                                               'fullyconnectedlayers': inputhparams.fullyconnectedlayers,
                                               'dropout': inputhparams.dropout,
                                               'recurrent': inputhparams.recurrent,
                                               'negsampling': inputhparams.negsampling,
                                               'cdf_tol':inputhparams.cdf_tol,
                                               'numsteps':inputhparams.numsteps
                                    })
                                 )
        '''
        self.save_hyperparameters(inputhparams)
        self.module = ARGVA(encoder, 
                            discriminator, 
                            node_decoder)
        self.numsteps = inputhparams.numsteps
        self.lr = inputhparams.learning_rate
        self.negsampling = inputhparams.negsampling
        self.numnodespergraph = numnodespergraph
        self.cdf_tol = inputhparams.cdf_tol
        
        # cache for generated latents
        self.generated_latents = None
        
    def VGAE_step(self,
                  batch,
                  name):
        #store original graph descriptors
        x = batch.x.clone()
        ea = batch.edge_attr.clone()
        ei = batch.edge_index.clone()
        ea[torch.isnan(ea)] = 0
        x[torch.isnan(x)] = 0
        #variational encoding
        self.generated_latents = self.module.encode(batch)
        #restore input edge features and node features
        batch.edge_index = ei.long()
        batch.edge_attr = ea
        batch.x = x
            
        graph_recon_loss, node_recon_loss = self.module.VGAE.recon_loss(self.generated_latents,
                                                                        batch,
                                                                        negsampling = self.negsampling,
                                                                        nodespergraph = self.numnodespergraph,
                                                                        cdf_tol = self.cdf_tol
                                                                       )
            
        kl_loss = (1/batch.num_nodes)*self.module.kl_loss()
        reg_loss = self.module.reg_loss(self.generated_latents)
            
        self.log(f"{name}_graph_recon_loss",
                 graph_recon_loss)
        self.log(f"{name}_node_recon_loss",
                 node_recon_loss)
        self.log(f"{name}_kl_loss",
                 kl_loss)
        self.log(f"{name}_reg_loss",
                 reg_loss)
            
        vgae_loss = node_recon_loss + graph_recon_loss + kl_loss + reg_loss
            
        self.log(f"{name}_VGAE_loss",vgae_loss)
            
        tqdm_dict = {f'{name}_VGAE_loss': vgae_loss}
        output = OrderedDict({
                'loss': vgae_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
        return output
    
    def discriminator_step(self,
                           batch,
                           name):
        
        disc_loss = self.module.discriminator_loss(self.generated_latents)
        self.log(f"{name}_disc_loss",
                 disc_loss)
            
        tqdm_dict = {f'{name}_d_loss': disc_loss}
        output = OrderedDict({
                'loss': disc_loss,
                'progress_bar': tqdm_dict,
                'log': tqdm_dict
            })
        return output
        
    def training_step(self, 
                      batch, 
                      batch_idx, 
                      optimizer_idx):
        self.module.train()
        if optimizer_idx == 0:
            output = self.VGAE_step(batch,
                                    'train')
            return output
        else:
            output = self.discriminator_step(batch,
                                             'train')
            return output

    def validation_step(self, 
                        batch, 
                        batch_idx):
        self.module.eval()
        VGAE_output = self.VGAE_step(batch,
                                    'val')
        disc_output = self.discriminator_step(batch,
                                              'val') 
        return VGAE_output

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
    