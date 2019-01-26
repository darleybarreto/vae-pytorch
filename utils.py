import csv
import torch
import numpy as np
from torch import optim
from torch.nn import functional as F
from torchvision.utils import make_grid

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def cifar_reshape(img):
    # transpose numpy array to the PIL format, i.e., Channels x W x H
    out = np.transpose(img, (1,2,0))
    return (out * 255).astype(np.uint8)

def save_attn_map(attns, imgs, info_saving):
        path, dataset, size = info_saving
        # append images and attention weights
        # imgs = imgs.clone()
        # imgs = torch.stack([self.unorm(im) for im in imgs]) #unormalize images
        img = torch.cat((imgs, attns), 0)
        # making a grid of two columns, images in one and attention in the other
        grid = make_grid(img,nrow=attns.size(0),padding=8)
        npimg = grid.detach().cpu().numpy() # to numpy array

        fig, ax = plt.subplots(figsize = tuple(size))
        ax.axis("off")
        ax.imshow(cifar_reshape(npimg))

        fig.savefig("{}.pdf".format(path),bbox_inches='tight')
        plt.close(fig)

def vae_recons_loss(recon_x, x):
    MSE = F.mse_loss(recon_x, x, reduction='sum')
    return MSE

class Model(object):
    def __init__(self, VAE, z_dim):
        self.vae = VAE(z_dim=z_dim)
        self.vae_opptimizer = optim.Adam(self.vae.parameters(),lr=1e-3,weight_decay=5e-4)

        if torch.cuda.is_available():
            self.vae = self.vae.cuda()

    def __call__(self, imgs):
        return self.vae(imgs)

    def zero_grad(self):
        self.vae_opptimizer.zero_grad()

    def step(self):
        self.vae_opptimizer.step()

    def train(self):
        self.vae.train()

class Logger(object):
    """
    :path:      path to save the file logger
    :header:    string list of the csv header
    """
    def __init__(self, path: str, header: list):
        self.log_file = open(path, 'w')
        self.logger = csv.writer(self.log_file, delimiter='\t')

        self.logger.writerow(header)
        self.header = header

    def close(self):
        self.log_file.close()

    def log(self, values):
        write_values = []
        for col in self.header:
            assert col in values
            write_values.append(values[col])
        self.logger.writerow(write_values)
        self.log_file.flush()

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count