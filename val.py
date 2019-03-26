import os
import torch
from utils import save_attn_map

def val(epoch, model, generator, compute_vae, metrics, folders, opt, logger):
    recons_meter, kl_meter, total = metrics
    models_folder, maps_folder = folders

    total_step: int = len(generator)
    itr: int = 0 # number of iteration
    total_log_p_x: int = 0

    model.eval()
    print("Evaluating:")
    log_size = torch.log(torch.Tensor([generator.batch_size]))

    with torch.no_grad():
        for batch_data in generator:  # doing this way we get more performance, since we generate one batch at a time
            imgs = batch_data[0]

            imgs_ = imgs.squeeze(0)

            if torch.cuda.is_available():
                imgs_ = imgs_.cuda(non_blocking=True)

            recon_batch, elbo = compute_vae(model, imgs_, metrics)

            log_p_x = torch.logsumexp(elbo, dim=0) - log_size
            total_log_p_x += log_p_x

            elbo = torch.sum(elbo)
            total.update(elbo.item(),imgs_.size(0))

            if itr == 0 and epoch % 50 == 0: # saving in each opt.attn_step batches
                save_attn = os.path.join(maps_folder, "val", "images_{}_{}".format(epoch, itr)), opt.dataset, (100,4)
                save_attn_map(recon_batch, imgs_, save_attn)         

            itr += 1
        
        total_log_p_x  = total_log_p_x.cpu().numpy().mean().sum()/len(generator)

        print("Recons ({})".format(recons_meter.avg),"+ KL ({})".format(kl_meter.avg), "= Full ({})".format(total.avg))
        print("Marginal Likelihood Estimation:",total_log_p_x)

        logger.log({'Epoch':'%d'%(epoch), "Recons": recons_meter.avg, \
                'KL': kl_meter.avg, "Full": recons_meter.avg +  kl_meter.avg, "MLikeli": total_log_p_x})

    return total.avg,total_log_p_x