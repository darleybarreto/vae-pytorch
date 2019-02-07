import os
import shutil
import argparse
import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torchvision import datasets as dsets, transforms

from models.gamma_vae import gamma_vae, compute_gamma
from models.normal_vae import gaussian_vae, compute_gaussian
from models.hierarchical import hierarchical_vae, compute_hierarchical

from utils import AverageMeter, Logger, Model, ReshapeTransform, save_attn_map, save_model

if torch.cuda.is_available(): torch.set_default_tensor_type('torch.cuda.FloatTensor')

parser = argparse.ArgumentParser()

parser.add_argument('--model', type=str, default="normal",
					help='VAE type (normal, gamma or hierarchical)')

parser.add_argument('--dataset', type=str, default="mnist",
					help='Dataset (cifar-10 or mnist)')

parser.add_argument('--epoches', type=int, default=1001,
					help='number of epoches')

parser.add_argument('--b_size', type=int, default=128,
					help='batch size')                    

parser.add_argument('--z_dim', type=int, default=4,
					help='size of the latent space')  

opt = parser.parse_args()

data_folder = os.path.join(".","data")
if not os.path.isdir(data_folder):
    os.makedirs(data_folder)

if opt.dataset == 'cifar-10':
    dataset = dsets.CIFAR10(root=os.path.join(".",'data'), train=True, 
        download=True, transform=transforms.Compose([transforms.ToTensor()]))

elif opt.dataset == 'mnist':
    dataset = dsets.MNIST(root=os.path.join(".",'data'), train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(), ReshapeTransform((-1,))]))

    if os.path.exists(os.path.join(".","data","raw")):
        shutil.rmtree(os.path.join(".","data","raw"))

t_generator = DataLoader(dataset, batch_size=opt.b_size, num_workers=8, shuffle=True)

if os.path.exists(os.path.join(data_folder,'cifar-10-python.tar.gz')):
    os.remove(os.path.join(data_folder,'cifar-10-python.tar.gz'))

assert opt.model in ['normal', 'gamma', 'hierarchical'], "Model {} is not supported.".format(opt.model)

recons_meter = AverageMeter()
kl_meter = AverageMeter()

metrics = [recons_meter, kl_meter]
logger_list = ["Epoch","Recons","KL","Full"]

if opt.model == 'normal':
    vae_model = gaussian_vae(opt.dataset)
    train_logger = Logger(os.path.join(data_folder,'train_normal.log'),logger_list)
    compute_vae = compute_gaussian

elif opt.model == 'gamma':
    vae_model = gamma_vae(opt.dataset)
    train_logger = Logger(os.path.join(data_folder,'train_gamma.log'),logger_list)
    compute_vae = compute_gamma

elif opt.model == 'hierarchical':
    logger_list = ["Epoch","Recons","KL Gamma", "KL Gaussian", "KL Final","Full"]

    kl_normal_meter = AverageMeter()
    kl_final_meter = AverageMeter()
    full_meter = AverageMeter()

    metrics += [kl_normal_meter, kl_final_meter, full_meter]

    vae_model = hierarchical_vae(opt.dataset)
    train_logger = Logger(os.path.join(data_folder,'train_hierarchical.log'),logger_list)
    compute_vae = compute_hierarchical

maps_folder = os.path.join(data_folder, "maps", opt.model)
if not os.path.isdir(maps_folder):
    os.makedirs(maps_folder)

models_folder = os.path.join(data_folder, "models")
if not os.path.isdir(models_folder):
    os.makedirs(models_folder)

print("{} model choosed.\n".format(opt.model))

vae = Model(vae_model,z_dim=opt.z_dim)
vae.train()

total_step: int = len(dataset)
best_loss = float("inf")
best_epoch = -1

for epoch in range(opt.epoches):

    for m in metrics:
        m.reset()

    print("Epoch", epoch)
    itr: int = 0 # number of iteration

    for imgs, _ in t_generator:  # doing this way we get more performance, since we generate one batch at a time
        vae.zero_grad()

        imgs_ = imgs.squeeze(0)

        if torch.cuda.is_available():
            imgs_ = imgs_.cuda(non_blocking=True)

        recon_batch, vae_loss = compute_vae(vae, imgs_, metrics)
        vae_loss.backward()
        vae.step()
        
        if itr == 0 and epoch % 50 == 0: # saving in each opt.attn_step batches
            save_attn = os.path.join(maps_folder, "images_{}_{}".format(epoch, itr)), opt.dataset, (100,4)
            save_attn_map(recon_batch, imgs_, save_attn)         

        itr += 1

    if recons_meter.avg < best_loss:
        best_loss = recons_meter.avg
        best_epoch = epoch
        is_best = True

    else:
        is_best = False
    
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
    
    if opt.model == 'hierarchical':
        print("Recons ({})".format(recons_meter.avg),"+ KL Gamma({})".format(kl_meter.avg),"+ KL Gaussian({})".format(kl_normal_meter.avg),\
            "+ KL Final({})".format(kl_final_meter.avg) , "= Full ({})".format(full_meter.avg))
        
        train_logger.log({'Epoch':'[%d/%d]'%(epoch,opt.epoches), "Recons": recons_meter.avg, \
                "KL Gamma":kl_meter.avg,"KL Gaussian":kl_normal_meter.avg, "KL Final": kl_final_meter.avg, "Full": full_meter.avg})
    
    else:
        print("Recons ({})".format(recons_meter.avg),"+ KL ({})".format(kl_meter.avg), "= Full ({})".format(recons_meter.avg + kl_meter.avg))
        
        train_logger.log({'Epoch':'[%d/%d]'%(epoch,opt.epoches), "Recons": recons_meter.avg, \
                'KL': kl_meter.avg, "Full": recons_meter.avg +  kl_meter.avg})

train_logger.close()