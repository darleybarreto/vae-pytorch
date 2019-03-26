import os
import csv
import torch
import shutil
import numpy as np
from glob import glob

import torch
from torch import optim
from torchvision.utils import make_grid

import urllib.request as request

from models.gamma_vae import gamma_vae
from models.normal_vae import gaussian_vae

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt

def binarized_mnist_fixed_binarization(DATASETS_DIR):
    """Get the binarized MNIST dataset and convert to hdf5.
    From https://github.com/altosaar/proximity_vi/blob/master/get_binary_mnist.py
    """

    subdatasets = ['train', 'valid', 'test']
    data = {}

    if not os.path.exists(DATASETS_DIR):
        os.makedirs(DATASETS_DIR)

        def lines_to_np_array(lines):
            return np.array([[int(i) for i in line.split()] for line in lines])

        for subdataset in subdatasets:
            filename = 'binarized_mnist_{}.amat'.format(subdataset)
            url = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/binarized_mnist_{}.amat'.format(
                subdataset)
            local_filename = os.path.join(DATASETS_DIR, filename)
            request.urlretrieve(url, local_filename)

            with open(os.path.join(DATASETS_DIR, filename)) as f:
                lines = f.readlines()
            
            os.remove(local_filename)
            data[subdataset] = lines_to_np_array(lines).astype('float32')
            np.savez_compressed(local_filename.split(".amat")[0],data=data[subdataset])

    else:
        for subdataset in subdatasets:
            filename = 'binarized_mnist_{}.npz'.format(subdataset)
            local_filename = os.path.join(DATASETS_DIR, filename)
            data[subdataset] = np.load(local_filename)

    return data['train']['data'], data['valid']['data'], data['test']['data']


def reshape(img):
    # transpose numpy array to the PIL format, i.e., Channels x W x H
    out = np.transpose(img, (1,2,0))
    return (out * 255).astype(np.uint8)

def save_attn_map(attns, imgs, info_saving):
    path, dataset, size = info_saving
    # append images and attention weights
    imgs = imgs.clone()

    if dataset in ['mnist','b_mnist']:
        attns = attns.view(-1,1,28,28)
        imgs = imgs.view(-1,1,28,28)
    
    # imgs = torch.stack([self.unorm(im) for im in imgs]) #unormalize images
    img = torch.cat((imgs, attns), 0)
    # making a grid of two columns, images in one and attention in the other
    grid = make_grid(img,nrow=attns.size(0),padding=8)
    npimg = grid.detach().cpu().numpy() # to numpy array

    fig, ax = plt.subplots(figsize = tuple(size))
    ax.axis("off")
    ax.imshow(reshape(npimg))

    fig.savefig("{}.pdf".format(path),bbox_inches='tight')
    plt.close(fig)


def save_model(internal_state, models_folder, is_best, epoch, model):
    checkpoints = sorted(glob(os.path.join(models_folder,'checkpoint_{}_*.pth.tar'.format(model))),key=lambda x: int(x.split("_")[-1].split(".")[0]))

    while len(checkpoints) > 1:
        first_check = checkpoints[0]
        os.remove(first_check)

        checkpoints = sorted(glob(os.path.join(models_folder,'checkpoint_{}_*.pth.tar'.format(model))),key=lambda x: int(x.split("_")[-1].split(".")[0]))
    
    del checkpoints
    file_name = os.path.join(models_folder, "checkpoint_{}_{}.pth.tar".format(model,epoch))
    torch.save(internal_state, file_name)

    if is_best:
        shutil.copyfile(file_name, os.path.join(models_folder, 'model_best_{}.pth.tar'.format(model)))

class ReshapeTransform:
    def __init__(self, new_size):
        self.new_size = new_size

    def __call__(self, img):
        return torch.reshape(img, self.new_size)

class Model(object):
    def __init__(self, VAE, z_dim):
        self.vae = VAE(z_dim=z_dim)
        self.vae_optimizer = optim.Adam(self.vae.parameters(),lr=1e-3,weight_decay=5e-4)
        
        if torch.cuda.is_available():
            self.vae = self.vae.cuda()

    def __call__(self, imgs):
        return self.vae(imgs)

    def zero_grad(self):
        self.vae_optimizer.zero_grad()

    def step(self):
        self.vae_optimizer.step()

    def train(self):
        self.vae.train()

    def eval(self):
        self.vae.eval()

    def sample(self, z):
        return self.vae.decode(z)

    def representation(self, x):
        encoded = self.vae.encode(x)
        z = self.vae.reparameterize(*encoded)
        return z.data

    @staticmethod
    def load(path: str, map_location='cuda'):
        checkpoint = torch.load(path,map_location=map_location)
        
        dataset = checkpoint['dataset']
        model = checkpoint['model']

        if model == 'normal':
            vae_model = gaussian_vae(dataset)
        else:
            vae_model = gamma_vae(dataset)

        self = Model(vae_model,checkpoint['z_dim'])

        self.vae.load_state_dict(checkpoint['model_vae_state_dict'])
        self.vae_optimizer.load_state_dict(checkpoint['optimizer_vae_state_dict'])

        return self

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