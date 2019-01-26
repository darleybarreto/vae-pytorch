import torch
from torch import nn
from torch.autograd import Variable

from torch.distributions.gamma import Gamma
from .vae import VAE

import sys
sys.path.append("..")
from utils import vae_recons_loss

def vae_gamma_loss(x, alpha, beta):
    loss = -torch.lgamma(alpha) - alpha * torch.log(alpha) \
        + (alpha - 1.) * torch.log(x) - x / beta
    return torch.sum(loss)

def compute_gamma(model, imgs_, metrics):
    recon_batch, z, alpha, beta = model(imgs_)
    recons_loss = vae_recons_loss(recon_batch, imgs_)

    sigma_negative_entropy_loss = vae_gamma_loss(z, alpha, beta)

    log_prior_loss = vae_gamma_loss(z, 1 + torch.ones(alpha.shape), torch.ones(alpha.shape))

    vae_loss = recons_loss + log_prior_loss - sigma_negative_entropy_loss

    metrics[0].update(recons_loss.item(),imgs_.size(0))
    metrics[1].update(sigma_negative_entropy_loss.item(),imgs_.size(0))
    metrics[2].update(log_prior_loss.item(),imgs_.size(0))
    metrics[3].update(vae_loss.item(),imgs_.size(0))

    return recon_batch, vae_loss

class GammaVAE(VAE):
    def __init__(self, h_dim=512, z_dim=512, gamma_shape = 8):
        super(GammaVAE, self).__init__(h_dim, z_dim)
        
        self.gamma_shape = gamma_shape

    def encode(self, x):
        fc1 = self._encode(x)    
        #alpha, beta
        return self.softplus(self.fc41(fc1)), self.softplus(self.fc42(fc1))

    def reparameterize(self, alpha, beta):
        """
        :alpha:  is the shape/concentration
        :beta:   is the rate/(1/scale)
        """
        # sample the \hat{z} ~ Gamma(shape + B, 1.) to guarantee acceptance
        new_alpha = alpha.clone()
        new_alpha = Variable(new_alpha + self.gamma_shape,requires_grad=False)
        z_hat = Gamma(new_alpha, torch.ones(alpha.shape)).sample()
        # compute the epsilon corresponding to \hat{z}; this epsilon is 'accepted'
        # \epsilon = h_inverse(z_tilde; shape + B)
        eps = self.compute_h_inverse(z_hat, alpha + self.gamma_shape)
        # now compute z_tilde = h(epsilon, alpha + gamma_shape)
        z_tilde = self.compute_h(eps, alpha + self.gamma_shape)
        return z_tilde/beta

    def compute_h(self, eps, alpha):
        return (alpha - 1. / 3.) * (1. + eps / torch.sqrt(9. * alpha - 3.))**3

    def compute_h_inverse(self, z, alpha):
        return torch.sqrt(9. * alpha - 3.) * ((z / (alpha - 1. / 3.))**(1. / 3.) - 1.)

    def forward(self, x):
        alpha, beta = self.encode(x)
        z = self.reparameterize(alpha, beta)
        return self.decode(z), z, alpha, beta