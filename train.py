import os
import torch
from utils import save_attn_map

def train(epoch, model, generator, compute_vae, metrics, folders, opt, logger):
    recons_meter, kl_meter, total = metrics
    models_folder, maps_folder = folders

    total_step: int = len(generator)
    itr: int = 0 # number of iteration
    
    model.train()
    print("Training:")

    for batch_data in generator:  # doing this way we get more performance, since we generate one batch at a time
        imgs = batch_data[0]
        model.zero_grad()

        imgs_ = imgs.squeeze(0)

        if torch.cuda.is_available():
            imgs_ = imgs_.cuda(non_blocking=True)

        recon_batch, vae_loss = compute_vae(model, imgs_, metrics)
        vae_loss = torch.sum(vae_loss)
        
        vae_loss.backward()
        model.step()
        
        total.update(vae_loss.item(),imgs_.size(0))

        if itr == 0 and epoch % 50 == 0: # saving in each opt.attn_step batches
            save_attn = os.path.join(maps_folder, "train", "images_{}_{}".format(epoch, itr)), opt.dataset, (100,4)
            save_attn_map(recon_batch, imgs_, save_attn)         

        itr += 1
    
    print("Recons ({})".format(recons_meter.avg),"+ KL ({})".format(kl_meter.avg), "= Full ({})".format(total.avg))
    
    logger.log({'Epoch':'%d'%(epoch), "Recons": recons_meter.avg, \
            'KL': kl_meter.avg, "Full": total.avg})