import torch
from torch import nn
from torch.autograd import Variable
from torch.distributions.gamma import Gamma
from .vae import choose_vae, recons_loss

def evaluate_gamma_dist(x, alpha, beta):
    loss = -torch.lgamma(alpha) - alpha * torch.log(alpha) \
        + (alpha - 1.) * torch.log(x) - x / beta
    
    return torch.sum(loss)

def I_function(a,b,c,d):
    return - c * d / a - b * torch.log(a) - torch.lgamma(b) + (b-1)*(torch.digamma(d) + torch.log(c))

def vae_gamma_kl_loss(a,b,c,d):
    """
    https://stats.stackexchange.com/questions/11646/kullback-leibler-divergence-between-two-gamma-distributions
    b and d are Gamma shape parameters and
    a and c are scale parameters.
    (All, therefore, must be positive.)
    """
    
    a = 1/a
    c = 1/c
    losses = I_function(c,d,c,d) - I_function(a,b,c,d)

    return torch.sum(losses)

def compute_gamma(model, imgs_, metrics):
    recon_batch, z, alpha, beta = model(imgs_)

    likelihood = recons_loss(recon_batch, imgs_)

    # posterior = evaluate_gamma_dist(z, alpha, beta)
    # prior = evaluate_gamma_dist(z, 1 + torch.ones(alpha.shape), torch.ones(alpha.shape))
    # vae_loss = likelihood + prior - posterior

    kl_gamma = vae_gamma_kl_loss(alpha, beta, torch.Tensor([2.]), torch.Tensor([1.])) #prior p(z|alpha,beta)) = p(z|(2,1)) 
    vae_loss = likelihood + kl_gamma

    metrics[0].update(likelihood.item(),imgs_.size(0))
    metrics[1].update(kl_gamma.item(),imgs_.size(0))

    return recon_batch, vae_loss

def gamma_vae(vae_name: str):
    VAE = choose_vae(vae_name)

    class GammaVAE(VAE):
        def __init__(self, h_dim=512, z_dim=32, gamma_shape = 8):
            super(GammaVAE, self).__init__(h_dim, z_dim)
            
            self.gamma_shape = gamma_shape

        def encode(self, x):
            fc1 = self._encode(x)    
            #alpha, beta
            return self.softplus(self.fc21(fc1)), self.softplus(self.fc22(fc1))

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

    return GammaVAE