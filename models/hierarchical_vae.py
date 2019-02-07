import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions.gamma import Gamma
from .vae import choose_vae

import sys
sys.path.append("..")

from utils import recons_loss

from .gamma_vae import vae_gamma_kl_loss
from .normal_vae import vae_gaussian_kl_loss

def compute_hierarchical(model, imgs_, metrics):
    recon_batch, z, mu_hat, logvar_hat, alpha, beta, mu, sigma = model(imgs_)

    likelihood = recons_loss(recon_batch, imgs_)

    kl_gamma = vae_gamma_kl_loss(alpha, beta, torch.Tensor([2.]), torch.Tensor([1.])) #prior p(z|alpha,beta)) = p(z|(2,1)) 

    kl_normal = vae_gaussian_kl_loss(mu_hat, logvar_hat)

    kl_final = vae_gaussian_kl_loss(mu, sigma)
    final_vae_loss = likelihood + kl_gamma + kl_normal + kl_final

    metrics[0].update(likelihood.item(),imgs_.size(0))
    metrics[1].update(kl_gamma.item(),imgs_.size(0))
    metrics[2].update(kl_normal.item(),imgs_.size(0))
    metrics[3].update(kl_final.item(),imgs_.size(0))
    metrics[4].update(final_vae_loss.item(),imgs_.size(0))

    return recon_batch, final_vae_loss

def hierarchical_vae(vae_name: str):
    VAE = choose_vae(vae_name)

    class HierarchicalVAE(VAE):
        def __init__(self, h_dim=512, z_dim=32, gamma_shape = 8):
            super(HierarchicalVAE, self).__init__(h_dim, z_dim)
            
            self.gamma_shape = gamma_shape

        def encode(self, x):
            fc1 = self._encode(x)    
            #mu, simg,a alpha, beta
            return self.fc41(fc1), self.softplus(self.fc42(fc1)), self.softplus(self.fc43(fc1)), self.softplus(self.fc44(fc1))

        def reparameterize_gamma(self, alpha, beta):
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

        def reparameterize_gaussian(self, mu, logvar):
            if self.training:
                std = logvar.mul(0.5).exp_()  # type: Variable
                eps = Variable(std.data.new(std.size()).normal_(),requires_grad=False)

                return eps.mul(std).add_(mu)

            else:
                return mu

        def compute_h(self, eps, alpha):
            return (alpha - 1. / 3.) * (1. + eps / torch.sqrt(9. * alpha - 3.))**3

        def compute_h_inverse(self, z, alpha):
            return torch.sqrt(9. * alpha - 3.) * ((z / (alpha - 1. / 3.))**(1. / 3.) - 1.)

        def forward(self, x):
            mu_hat, logvar_hat, aplha, beta = self.encode(x)

            mu = self.reparameterize_gaussian(mu_hat, logvar_hat)
            sigma = self.reparameterize_gamma(aplha, beta)

            z = self.reparameterize_gaussian(mu, sigma)

            return self.decode(z), z, mu_hat, logvar_hat, aplha, beta, mu, sigma

    return HierarchicalVAE