import torch
from torch import Tensor
from torch.nn import L1Loss, Parameter
from torch import sigmoid
from torch.distributions.beta import Beta

from scipy.special import betainc

from ..layers.utils import reset
from torch_geometric.utils import (batched_negative_sampling, remove_self_loops,
                                   add_self_loops)

EPS = 1e-15
MAX_LOGSTD = 10
MAX_LOGPROB = 50
TOL = 0.001
CHUNKSIZE = 1000
NODESPERGRAPH = 51

class BetaPriorDecoder(torch.nn.Module):
    r"""Beta prior decoder"""
    def __init__(self,
                 logprecision=torch.tensor([1.0]),
                 loggamma=torch.tensor([1.0]),
                 logN = torch.tensor([1.0])
                ):
        super(BetaPriorDecoder,self).__init__()
        self.logprecision = Parameter(logprecision)
        self.loggamma = Parameter(loggamma)
        self.logN = Parameter(logN)
        
    def get_beta_params(self,
                        z,
                        edge_index=None
                       ):
        if edge_index is None:
            shape = z.shape[0]
            diff = abs(torch.arange(shape)-torch.arange(shape).view(shape,1))+1
            p = sigmoid(torch.matmul(z, z.t()))
        else:
            diff = edge_index.diff(dim=0).abs()+1
            p = sigmoid((z[edge_index[0]]*z[edge_index[1]]).sum(dim=1))
            
        diff = torch.pow(diff,
                         -torch.exp(self.loggamma))
        prior_alpha = diff*torch.exp(self.logprecision)
        prior_beta = (1-diff)*torch.exp(self.logprecision)
        
        posterior_alpha = prior_alpha + p*torch.exp(self.logN)
        posterior_beta = prior_beta + ((1-p)*torch.exp(self.logN))
        
        return posterior_alpha.squeeze()+EPS, posterior_beta.squeeze()+EPS 
    
    def get_means(self,
                  z,
                  edge_index = None
                 ):
        posterior_alpha, posterior_beta = self.get_beta_params(z, 
                                                               edge_index)
        
        means = posterior_alpha/(posterior_alpha+posterior_beta)
        
        return means
        
    def forward(self, 
                z, 
                edge_index,
                edge_attr,
                chunksize = CHUNKSIZE,
                tol = TOL,
                cdf_tol = None
               ):
        r"""Decodes the latent variables :obj:`z` into edge probabilities for
        the given node-pairs :obj:`edge_index`.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        posterior_alpha, posterior_beta = self.get_beta_params(z,
                                                               edge_index)
        
        edge_attr = edge_attr.squeeze()
        edge_attr = torch.clamp(edge_attr,
                                min=tol,
                                max=1-tol)

        lims = torch.arange(0,
                            edge_attr.shape[0],
                            chunksize).cuda()
        edge_logprobs = []
        if cdf_tol is not None:
            lims_down = torch.clone(Tensor.cpu(torch.clamp(edge_attr - cdf_tol, 
                                                           min = tol)
                                              )
                                   ).detach()
            lims_up = torch.clone(Tensor.cpu(torch.clamp(edge_attr + cdf_tol,
                                                         max = 1-tol)
                                            )
                                 ).detach()
            posterior_alpha = Tensor.cpu(posterior_alpha).detach()
            posterior_beta = Tensor.cpu(posterior_beta).detach()
        for lim in lims[:-1]:
            if cdf_tol is not None:
                cdf_down = betainc(posterior_alpha[lim:lim+chunksize],
                                   posterior_beta[lim:lim+chunksize],
                                   lims_down[lim:lim+chunksize])
                cdf_up = betainc(posterior_alpha[lim:lim+chunksize],
                                 posterior_beta[lim:lim+chunksize],
                                 lims_up[lim:lim+chunksize])
                prob = cdf_up-cdf_down
                edge_logprobs.append(-torch.log(prob))
            else:
                edge_logprobs.append(-Beta(posterior_alpha[lim:lim+chunksize],
                                           posterior_beta[lim:lim+chunksize]).log_prob(edge_attr[lim:lim+chunksize])
                                    )
                
        if cdf_tol is not None:
            cdf_down = betainc(posterior_alpha[lims[-1]:],
                               posterior_beta[lims[-1]:],
                               lims_down[lims[-1]:])
            cdf_up = betainc(posterior_alpha[lims[-1]:],
                             posterior_beta[lims[-1]:],
                             lims_up[lims[-1]:])
            prob = cdf_up-cdf_down
            edge_logprobs.append(-torch.log(prob))
        else:
            edge_logprobs.append(-Beta(posterior_alpha[lims[-1]:],
                                       posterior_beta[lims[-1]:]).log_prob(edge_attr[lims[-1]:])
                                )
        
        edge_logprobs = torch.clamp(torch.cat(edge_logprobs).cuda(),
                                    max = MAX_LOGPROB)
                                    
        return edge_logprobs

    def forward_all(self, 
                    z,
                    edge_index,
                    edge_attr
                   ):
        r"""Decodes the latent variables :obj:`z` into a probabilistic dense
        adjacency matrix.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            sigmoid (bool, optional): If set to :obj:`False`, does not apply
                the logistic sigmoid function to the output.
                (default: :obj:`True`)
        """
        posterior_alpha, posterior_beta = self.get_beta_params(z,
                                                               edge_index)
        
        adj = sparse_coo_tensor(edge_index, 
                                edge_attr, 
                                [shape,shape]).to_dense()
        
        edge_logprobs = -Beta(posterior_alpha,
                              posterior_beta).log_prob(adj)
        
        return edge_logprobs


class GAE(torch.nn.Module):
    r"""The Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper based on user-defined encoder and decoder models.
    Args:
        encoder (Module): The encoder module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, 
                 encoder, 
                 node_decoder=None
                ):
        super(GAE, self).__init__()
        self.encoder = encoder
        self.node_decoder = node_decoder
        self.edge_decoder = BetaPriorDecoder()
        GAE.reset_parameters(self)

    def reset_parameters(self):
        reset(self.encoder)
        reset(self.node_decoder)
        reset(self.edge_decoder)

    def encode(self, *args, **kwargs):
        r"""Runs the encoder and computes node-wise latent variables."""
        return self.encoder(*args, **kwargs)

    def edge_decode(self, *args, **kwargs):
        r"""Runs the edge decoder and computes edge probabilities."""
        return self.edge_decoder(*args, **kwargs)
    
    def node_decode(self, *args, **kwargs):
        r"""Runs the node decoder and computes node reconstructions."""
        self.node_decoder(*args, **kwargs)

    def recon_loss(self, 
                   z,
                   batch,
                   negsampling = True,
                   nodespergraph = NODESPERGRAPH,
                   negfactor = 20,
                   cdf_tol = None
                  ):
        r"""Given latent variables :obj:`z`, computes the binary cross
        entropy loss for positive edges :obj:`pos_edge_index` and negative
        sampled edges.
        Args:
            x (Tensor): The input space :math:`\mathbf{Z}`.
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to train against.
            neg_edge_index (LongTensor, optional): The negative edges to train
                against. If not given, uses negative sampling to calculate
                negative edges. (default: :obj:`None`)
        """

        pos_edge_loss = self.edge_decoder(z,
                                          batch.edge_index,
                                          batch.edge_attr,
                                          cdf_tol = cdf_tol
                                         ).mean()

        if negsampling:
            # Do not include self-loops in negative samples
            pos_edge_index, _ = remove_self_loops(batch.edge_index)
            pos_edge_index, _ = add_self_loops(batch.edge_index)
            
            negsamples = batched_negative_sampling(batch.edge_index, 
                                                   batch.batch, 
                                                   num_neg_samples=int((nodespergraph**2/negfactor
                                                                       )
                                                                      )
                                                  ).cuda()
        
            neg_edge_loss = self.edge_decoder(z,
                                              negsamples,
                                              torch.full((negsamples.size(1),), 
                                                         0.0).cuda(),
                                              cdf_tol
                                             ).mean()
        else:
            neg_edge_loss = 0.0
        
        if self.node_decoder is not None:
            x = batch.x
            batch.x = z
            pred = self.node_decoder(batch)
            criterion = L1Loss()
            node_loss = criterion(pred,
                                  x)
        else:
            node_loss = 0.0
        
        edge_loss = pos_edge_loss + neg_edge_loss
        return edge_loss, node_loss

    def test_edges(self,
                   z, 
                   batch):
        r"""Given latent variables :obj:`z`, positive edges
        :obj:`pos_edge_index` and negative edges :obj:`neg_edge_index`,
        computes the L1loss of the predicted edges vs the real edges.
        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
            pos_edge_index (LongTensor): The positive edges to evaluate
                against.
            neg_edge_index (LongTensor): The negative edges to evaluate
                against.
        """
        # Do not include self-loops in negative samples
        pos_edge_index, _ = remove_self_loops(batch.edge_index)
        pos_edge_index, _ = add_self_loops(batch.edge_index)
        
        neg_edge_index = negative_sampling(batch.edge_index, 
                                           z.size(0))
        
        neg_y = z.new_zeros(neg_edge_index.size(1))
        y = torch.cat([batch.edge_attr, 
                       neg_y], 
                      dim=0)

        pos_pred = self.edge_decoder.get_means(z,
                                               batch.edge_index)
        neg_pred = self.edge_decoder.get_means(z,
                                               neg_edge_index)
        
        pred = torch.cat([pos_pred, 
                          neg_pred], 
                         dim=0)

        y, pred = y.detach().cpu().numpy(), pred.detach().cpu().numpy()

        return L1Loss(pred,
                      y)


class VGAE(GAE):
    r"""The Variational Graph Auto-Encoder model from the
    `"Variational Graph Auto-Encoders" <https://arxiv.org/abs/1611.07308>`_
    paper.
    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, 
                 encoder, 
                 node_decoder=None
                ):
        super(VGAE, self).__init__(encoder, 
                                   node_decoder)

    def reparametrize(self, mu, logstd):
        if self.training:
            return mu + torch.randn_like(logstd) * torch.exp(logstd)
        else:
            return mu

    def encode(self, *args, **kwargs):
        """"""
        self.__mu__, self.__logstd__ = self.encoder(*args, **kwargs)
        self.__logstd__ = self.__logstd__.clamp(max=MAX_LOGSTD)
        z = self.reparametrize(self.__mu__, self.__logstd__)
        return z

    def kl_loss(self, mu=None, logstd=None):
        r"""Computes the KL loss, either for the passed arguments :obj:`mu`
        and :obj:`logstd`, or based on latent variables from last encoding.
        Args:
            mu (Tensor, optional): The latent space for :math:`\mu`. If set to
                :obj:`None`, uses the last computation of :math:`mu`.
                (default: :obj:`None`)
            logstd (Tensor, optional): The latent space for
                :math:`\log\sigma`.  If set to :obj:`None`, uses the last
                computation of :math:`\log\sigma^2`.(default: :obj:`None`)
        """
        mu = self.__mu__ if mu is None else mu
        logstd = self.__logstd__ if logstd is None else logstd.clamp(
            max=MAX_LOGSTD)
        return -0.5 * torch.mean(
            torch.sum(1 + 2 * logstd - mu**2 - logstd.exp()**2, dim=1))
    
    
class ARGA(GAE):
    r"""The Adversarially Regularized Graph Auto-Encoder model from the
    `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.
    paper.

    Args:
        encoder (Module): The encoder module.
        discriminator (Module): The discriminator module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, 
                 encoder, 
                 discriminator, 
                 node_decoder=None
                ):
        super(ARGA, self).__init__(encoder, 
                                   node_decoder
                                  )
        self.discriminator = discriminator
        reset(self.discriminator)

    def reset_parameters(self):
        super(ARGA, self).reset_parameters()
        reset(self.discriminator)


    def reg_loss(self, z):
        r"""Computes the regularization loss of the encoder.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(z))
        real_loss = -torch.log(real + EPS).mean()
        return real_loss


    def discriminator_loss(self, z):
        r"""Computes the loss of the discriminator.

        Args:
            z (Tensor): The latent space :math:`\mathbf{Z}`.
        """
        real = torch.sigmoid(self.discriminator(torch.randn_like(z)))
        fake = torch.sigmoid(self.discriminator(z.detach()))
        real_loss = -torch.log(real + EPS).mean()
        fake_loss = -torch.log(1 - fake + EPS).mean()
        return real_loss + fake_loss



class ARGVA(ARGA):
    r"""The Adversarially Regularized Variational Graph Auto-Encoder model from
    the `"Adversarially Regularized Graph Autoencoder for Graph Embedding"
    <https://arxiv.org/abs/1802.04407>`_ paper.
    paper.

    Args:
        encoder (Module): The encoder module to compute :math:`\mu` and
            :math:`\log\sigma^2`.
        discriminator (Module): The discriminator module.
        decoder (Module, optional): The decoder module. If set to :obj:`None`,
            will default to the
            :class:`torch_geometric.nn.models.InnerProductDecoder`.
            (default: :obj:`None`)
    """
    def __init__(self, 
                 encoder, 
                 discriminator, 
                 node_decoder=None
                ):
        super(ARGVA, self).__init__(encoder, 
                                    discriminator, 
                                    node_decoder
                                   )
        self.VGAE = VGAE(encoder, 
                         node_decoder
                        )

    @property
    def __mu__(self):
        return self.VGAE.__mu__

    @property
    def __logstd__(self):
        return self.VGAE.__logstd__

    def reparametrize(self, mu, logstd):
        return self.VGAE.reparametrize(mu, logstd)


    def encode(self, *args, **kwargs):
        """"""
        return self.VGAE.encode(*args, **kwargs)


    def kl_loss(self, mu=None, logstd=None):
        return self.VGAE.kl_loss(mu, logstd)