import torch
import numpy as np
import torch.nn.functional as F

from methods.utils import create_window

# KL Divergence loss
def kl_loss(mu, logvar):
    kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return kl_divergence


# Log barrier loss
'https://github.com/jusiro/constrained_anomaly_segmentation/blob/main/code/models/models.py'
def log_barrier(z, t=5):

    # Only one value
    if z.shape[0] == 1:

        if z <= - 1 / t ** 2:
            log_barrier_loss = - torch.log(-z) / t
        else:
            log_barrier_loss = t * z + -np.log(1 / (t ** 2)) / t + 1 / t

    # Constrain over multiple values
    else:
        log_barrier_loss = torch.tensor(0).cuda().float()
        for i in np.arange(0, z.shape[0]):
            zi = z[i, 0]
            if zi <= - 1 / t ** 2:
                log_barrier_loss += - torch.log(-zi) / t
            else:
                log_barrier_loss += t * zi + -np.log(1 / (t ** 2)) / t + 1 / t

    return log_barrier_loss


# SSIM loss
'VTL-AE paper: https://arxiv.org/abs/2106.06716'

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)
        
def ssim_loss(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)