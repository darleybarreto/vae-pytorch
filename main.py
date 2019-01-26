import os
import argparse
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, sampler
from torchvision import datasets as dsets, transforms

from models.gamma_vae import GammaVAE, compute_gamma
from models.normal_vae import GaussianVAE, compute_gaussian
from utils import AverageMeter, Logger, Model, save_attn_map

if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="normal",
					help='VAE type (normal or gamma)')

parser.add_argument('--epoches', type=int, default=10000,
					help='number of epoches')

opt = parser.parse_args()

data_folder = os.path.join(".","data")
if not os.path.isdir(data_folder):
    os.makedirs(data_folder)

dataset = dsets.CIFAR10(root='./data/', train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
t_generator = DataLoader(dataset, batch_size=512, num_workers=8, shuffle=True)

if os.path.exists(os.path.join(data_folder,'cifar-10-python.tar.gz')):
    os.remove(os.path.join(data_folder,'cifar-10-python.tar.gz'))

assert opt.model in ['normal', 'gamma'], "Model {} is not supported.".format(opt.model)

recons_meter = AverageMeter()
hat_meter = AverageMeter()

metrics = [recons_meter, hat_meter]

if opt.model == 'normal':
    vae_model = GaussianVAE
    train_logger = Logger(os.path.join(data_folder,'train_normal.log'),["Epoch","Recons","KL","Full"])
    compute_vae = compute_gaussian

elif opt.model == 'gamma':
    vae_model = GammaVAE
    train_logger = Logger(os.path.join(data_folder,'train_gamma.log'),["Epoch","Recons", "Prior","Entropy","Full"])
    compute_vae = compute_gamma

    prior_meter = AverageMeter()
    full_meter = AverageMeter()
    metrics += [prior_meter, full_meter]

maps_folder = os.path.join(data_folder, "maps", opt.model)
if not os.path.isdir(maps_folder):
    os.makedirs(maps_folder)

print("{} model choosed.".format(opt.model))

vae = Model(vae_model,z_dim=20)
vae.train()

total_step: int = len(dataset)

for epoch in range(opt.epoches):

    for m in metrics:
        m.reset()

    print("Epoch", epoch)
    itr: int = 0 # number of iteration

    for imgs, _ in t_generator:  # doing this way we get more performance, since we generate one batch at a time
        # print("Iteration {} of {}.".format(itr,total_step//512))
        if itr == 0 and epoch % 10 == 0: # saving in each opt.attn_step batches
            save_attn = os.path.join(maps_folder, "images_{}_{}".format(epoch, itr)), 'cifar-10', (100,4)
        else:
            save_attn = None

        vae.zero_grad()

        imgs_ = imgs.squeeze(0)

        if torch.cuda.is_available():
            imgs_ = imgs_.cuda(non_blocking=True)

        imgs_ = Variable(imgs_,requires_grad=True)
        
        recon_batch, vae_loss = compute_vae(vae, imgs_, metrics)

        vae_loss.backward()
        vae.step()

        if not save_attn is None:
            save_attn_map(recon_batch, imgs_, save_attn)

        itr += 1
    
    if opt.model == 'gamma':
        print("Recons ({})".format(recons_meter.avg),"+ Prior ({})".format(prior_meter.avg),"- Entropy ({})".format(hat_meter.avg), "= Full ({})".format(full_meter.avg))
        train_logger.log({'Epoch':'[%d/%d]'%(epoch,opt.epoches), "Recons": recons_meter.avg, \
                "Prior":prior_meter.avg,"Entropy":hat_meter.avg, "Full": full_meter.avg})
    
    else:
        print("Recons ({})".format(recons_meter.avg),"+ KL ({})".format(hat_meter.avg), "= Full ({})".format(recons_meter.avg + hat_meter.avg))
        train_logger.log({'Epoch':'[%d/%d]'%(epoch,opt.epoches), "Recons": recons_meter.avg, \
                'KL': hat_meter.avg, "Full": recons_meter.avg +  hat_meter.avg})

train_logger.close()