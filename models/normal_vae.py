# import math

import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

from .vae import choose_vae

import sys
sys.path.append("..")

from utils import recons_loss

def vae_gaussian_kl_loss(mu, logvar):
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return KLD

# def evaluate_gaussian_dist(loc, scale, z):
#     var = torch.pow(scale, 2)
#     return -0.5 * torch.log(2 * np.pi * var) - torch.pow(z - loc, 2) / (2 * var)

# bernoulli_log_lik =  nn.BCEWithLogitsLoss(reduction='none')

def compute_gaussian(model, imgs_, metrics):
    recon_batch, z, mu, sigma = model(imgs_)
    
    # We can derive two different, but equivalent losses:
    # log p(x) >= Expectation[q](log p(x,z) - log q(z|x))
    #          >= Expectation[q](log p(x|z) + log p(z) - log q(z|x))
    #          \approx Mean[log p(x|z) + log p(z) - log q(z|x)]
    
    # latent_size = z.size(1)
    # loc =  torch.zeros(latent_size)
    # scale = torch.ones(latent_size)

    # prior = evaluate_gaussian_dist(loc, scale, z).sum(-1, keepdim=True) # log p(z)
    # likelihood = -bernoulli_log_lik(recon_batch, imgs_).sum(-1, keepdim=True) # log p(x|z)
    # posterior = evaluate_gaussian_dist(mu, sigma, z).sum(-1) # approximation of the posteior q(z|x)

    # joint = likelihood + prior # log p(x,z)

    # elbo = (joint - posterior).mean(1)
    # vae_loss = -elbo.sum(0)
    
    # metrics[0].update(vae_loss.item())
    # metrics[1].update(elbo.detach().cpu().numpy().mean().item())

    # Or using the likelihood - DKL = log p(x|z) - DKL(q(z|x)||p(z))
    # The first term if the likelihood and the second the kl dissimilarity
    # we can work out the later based on the prior N(0,1)
    # see in bellow for the general formulation:
    # https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence#Multivariate_normal_distributions
    
    recons_loss = recons_loss(recon_batch, imgs_)
    kl_loss = vae_gaussian_kl_loss(mu, sigma)

    metrics[0].update(recons_loss.item())
    metrics[1].update(kl_loss.item())

    vae_loss = recons_loss + kl_loss

    return recon_batch, vae_loss

def gaussian_vae(vae_name: str):
    VAE = choose_vae(vae_name)

    class GaussianVAE(VAE):
        def __init__(self, h_dim=512, z_dim=32):
            super(GaussianVAE, self).__init__(h_dim, z_dim)

        def encode(self, x):
            fc1 = self._encode(x)
            #mu, sigma
            return self.fc21(fc1), self.softplus(self.fc22(fc1))

        def reparameterize(self, mu, logvar):
            """THE REPARAMETERIZATION IDEA:
            
            Commented and type annotated by Charl Botha <cpbotha@vxlabs.com>

            For each training sample

            - take the current learned mu, stddev for each of the ZDIMS
            dimensions and draw a random sample from that distribution
            - the whole network is trained so that these randomly drawn
            samples decode to output that looks like the input
            - which will mean that the std, mu will be learned
            *distributions* that correctly encode the inputs
            - due to the additional KLD term
            the distribution will tend to unit Gaussians

            Parameters
            ----------
            mu : [batche_size, ZDIMS] mean matrix
            logvar : [batche_size, ZDIMS] variance matrix

            Returns
            -------

            During training random sample from the learned ZDIMS-dimensional
            normal distribution; during inference its mean.

            """

            if self.training:
                # multiply log variance with 0.5, then in-place exponent
                # yielding the standard deviation
                std = logvar.mul(0.5).exp_()  # type: Variable
                # - std.data is the [batche_size,ZDIMS] tensor that is wrapped by std
                # - so eps is [batche_size,ZDIMS] with all elements drawn from a mean 0
                #   and stddev 1 normal distribution that is batche_size samples
                #   of random ZDIMS-float vectors
                eps = Variable(std.data.new(std.size()).normal_(),requires_grad=False)
                # - sample from a normal distribution with standard
                #   deviation = std and mean = mu by multiplying mean 0
                #   stddev 1 sample with desired std and mu, see
                #   https://stats.stackexchange.com/a/16338
                # - so we have batche_size sets (the batch) of random ZDIMS-float
                #   vectors sampled from normal distribution with learned
                #   std and mu for the current input
                return eps.mul(std).add_(mu)

            else:
                # During inference, we simply spit out the mean of the
                # learned distribution for the current input.  We could
                # use a random sample from the distribution, but mu of
                # course has the highest probability.
                return mu

        def forward(self, x):
            mu_hat, logvar_hat = self.encode(x)
            z = self.reparameterize(mu_hat, logvar_hat)
            return self.decode(z), z, mu_hat, logvar_hat + 1e-5
            
            # mu_hat, logvar_hat, aplha, beta = self.encode(x)
            # mu = self.reparameterize(mu_hat, logvar_hat)
            # sigma = self.reparameterize(aplha, beta)

            # z = self.reparameterize(mu, sigma)

            # return self.decode(z), mu, sigma

    return GaussianVAE