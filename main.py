import os
import shutil
import argparse
import numpy as np

import torch
from torch.utils.data import DataLoader
from torchvision import datasets as dsets, transforms

from models.gamma_vae import gamma_vae, compute_gamma
from models.normal_vae import gaussian_vae, compute_gaussian

from utils import AverageMeter, Logger, Model, ReshapeTransform, save_model

from train import train
from val import val

if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="normal",
					help='VAE type (normal or gamma)')

parser.add_argument('--dataset', type=str, default="mnist",
					help='Dataset (cifar-10 or mnist)')

parser.add_argument('--epoches', type=int, default=1001,
					help='number of epoches')

parser.add_argument('--b_size', type=int, default=128,
					help='batch size')                    

parser.add_argument('--z_dim', type=int, default=4,
					help='size of the latent space')  

opt = parser.parse_args()

assert opt.model in ['normal', 'gamma'], "Model {} is not supported.".format(opt.model)

data_folder = os.path.join(".","data")
if not os.path.isdir(data_folder):
    os.makedirs(data_folder)

if opt.dataset == 'cifar-10':
    t_dataset = dsets.CIFAR10(root=os.path.join(".",'data'), train=True, 
        download=True, transform=transforms.Compose([transforms.ToTensor()]))

    v_dataset = dsets.CIFAR10(root=os.path.join(".",'data'), train=False, 
        download=True, transform=transforms.Compose([transforms.ToTensor()]))

    if os.path.exists(os.path.join(data_folder,'cifar-10-python.tar.gz')):
        os.remove(os.path.join(data_folder,'cifar-10-python.tar.gz'))

elif opt.dataset == 'mnist':
    t_dataset = dsets.MNIST(root=os.path.join(".",'data'), train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(), ReshapeTransform((-1,))]))

    v_dataset = dsets.MNIST(root=os.path.join(".",'data'), train=False, download=True,
        transform=transforms.Compose([transforms.ToTensor(), ReshapeTransform((-1,))]))

    if os.path.exists(os.path.join(".","data","raw")):
        shutil.rmtree(os.path.join(".","data","raw"))


t_generator = DataLoader(t_dataset, batch_size=opt.b_size, num_workers=8, shuffle=True)
v_generator = DataLoader(v_dataset, batch_size=opt.b_size, num_workers=8, shuffle=True)

recons_meter = AverageMeter()
kl_meter = AverageMeter()
taotal_meter = AverageMeter()

metrics = [recons_meter, kl_meter, taotal_meter]
logger_list = ["Epoch","Recons","KL","Full"]

suffix_logger = "{}_{}_{}_{}".format(opt.model,opt.dataset,opt.b_size,opt.z_dim)

train_logger = Logger(os.path.join(data_folder,'train_{}.log'.format(suffix_logger)),logger_list)
val_logger = Logger(os.path.join(data_folder,'val_{}.log'.format(suffix_logger)),logger_list)

if opt.model == 'normal':
    vae_model = gaussian_vae(opt.dataset)
    compute_vae = compute_gaussian

elif opt.model == 'gamma':
    vae_model = gamma_vae(opt.dataset)
    compute_vae = compute_gamma


maps_folder = os.path.join(data_folder, "maps", opt.model)
if not os.path.isdir(maps_folder):
    os.makedirs(os.path.join(maps_folder,"train"))
    os.makedirs(os.path.join(maps_folder,"val"))

models_folder = os.path.join(data_folder, "models")
if not os.path.isdir(models_folder):
    os.makedirs(models_folder)

print("{} model chosen.\n".format(opt.model))

vae = Model(vae_model,z_dim=opt.z_dim)

best_loss = float("inf")
best_epoch = -1

for epoch in range(opt.epoches):

    for m in metrics:
        m.reset()

    print("====== Epoch {} ======".format(epoch))
    train(epoch, vae, t_generator, compute_vae, metrics, (models_folder, maps_folder), opt, train_logger)
    vae_loss = val(epoch, vae, v_generator, compute_vae, metrics, (models_folder, maps_folder), opt, val_logger)
    
    is_best = False
    if vae_loss < best_loss:
        best_loss = vae_loss
        best_epoch = epoch
        is_best = True
       
    
    internal_state = {
        'model':opt.model,
        'dataset': opt.dataset,
        'z_dim': opt.z_dim,
        'current_epoch': epoch,
        'best_epoch': best_epoch,
        'best_loss': best_loss,
        'model_vae_state_dict': vae.vae.state_dict(),
        'optimizer_vae_state_dict': vae.vae_optimizer.state_dict()

    }

    save_model(internal_state, models_folder, is_best, epoch, opt.model)
    

train_logger.close()
val_logger.close()