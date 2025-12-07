import torch
import ctypes
import torch.nn as nn
from torch.autograd import Variable
from math import exp

# Initialize weight function
def initialize_weights(*models):
    for model in models:
        for module in model.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight)
                if module.bias is not None:
                    module.bias.data.zero_()
            elif isinstance(module, nn.BatchNorm2d):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


# Add noise to the input
def add_noise(latent, noise_type="gaussian", sd=0.2):
    """
    Adds noise to the input tensor.

    Args:
        latent (torch.Tensor): Input tensor.
        noise_type (str): Type of noise to add ("gaussian", "speckle", or "coarse").
        sd (float): Standard deviation for the noise.

    Returns:
        torch.Tensor: Tensor with added noise.
    """
    assert sd >= 0.0
    if noise_type == "gaussian":
        mean = 0.
        n = torch.distributions.Normal(torch.tensor([mean]), torch.tensor([sd]))
        noise = n.sample(latent.size()).squeeze(-1).to(latent.device)
        latent = latent + noise
        return latent

    if noise_type == "speckle":
        noise = torch.randn(latent.size()).to(latent.device)
        latent = latent + latent * noise
        return latent

    if noise_type == "coarse":
        shape = latent.shape

        scale = 16
        coarse_shape = [shape[0], shape[1], max(1, shape[2] // scale), max(1, shape[3] // scale)]
        coarse_noise = torch.randn(coarse_shape, device=latent.device) * sd

        noise = torch.nn.functional.interpolate(coarse_noise, size=shape[2:], mode='bilinear', align_corners=False)
        latent = latent + noise
        return latent

    if noise_type == "none":
        return latent

# Recycle bin
def recycle_bin():
    SHERB_NOCONFIRMATION = 0x00000001  # No mostrar confirmación
    SHERB_NOPROGRESSUI = 0x00000002    # No mostrar barra de progreso
    SHERB_NOSOUND = 0x00000004         # No hacer sonido al vaciar

    resultado = ctypes.windll.shell32.SHEmptyRecycleBinW(None, None, SHERB_NOCONFIRMATION | SHERB_NOPROGRESSUI | SHERB_NOSOUND)

    if resultado == 0:
        print("Papelera vaciada correctamente.")
    else:
        print(f"Error al vaciar la papelera. Código de error: {resultado}")


# Gaussian function for SSIM
def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()


# Create window for SSIM
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


# Gradient CAM
def grad_cam(activations, output, normalization='relu_min_max', avg_grads=False, norm_grads=False):
    def normalize(grads):
        l2_norm = torch.sqrt(torch.mean(torch.pow(grads, 2))) + 1e-5
        return grads * torch.pow(l2_norm, -1)

    # Obtain gradients
    gradients = torch.autograd.grad(output, activations, grad_outputs=None, retain_graph=True, create_graph=True,
                                    only_inputs=True, allow_unused=True)[0]

    # Normalize gradients
    if norm_grads:
        gradients = normalize(gradients)

    # pool the gradients across the channels
    if avg_grads:
        gradients = torch.mean(gradients, dim=[2, 3])
        # gradients = torch.nn.functional.softmax(gradients)
        gradients = gradients.unsqueeze(-1).unsqueeze(-1)

    # weight activation maps
    '''
    if 'relu' in normalization:
        GCAM = torch.sum(torch.relu(gradients * activations), 1)
    else:
        GCAM = gradients * activations
        if 'abs' in normalization:
            GCAM = torch.abs(GCAM)
        GCAM = torch.sum(GCAM, 1)
    '''
    GCAM = torch.mean(activations, 1)

    # Normalize CAM
    if 'sigm' in normalization:
        GCAM = torch.sigmoid(GCAM)
    if 'min' in normalization:
        norm_value = torch.min(torch.max(GCAM, -1)[0], -1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-3
        GCAM = GCAM - norm_value
    if 'max' in normalization:
        norm_value = torch.max(torch.max(GCAM, -1)[0], -1)[0].unsqueeze(-1).unsqueeze(-1) + 1e-3
        GCAM = GCAM * norm_value.pow(-1)
    if 'tanh' in normalization:
        GCAM = torch.tanh(GCAM)
    if 'clamp' in normalization:
        GCAM = GCAM.clamp(max=1)

    return GCAM