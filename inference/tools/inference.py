import torch
import numpy as np

from torch.amp import autocast

from tools.postprocessing import postprocessing, get_connected_components
from tools.mask_fusing import mask_fusing
from tools.save_results import save_results

def inference(model, volume, volume_key, output_path, device):  
    xs, xs_hat, diffs, diffs_post = [], [], [], []
    for i in range(volume.shape[-1]):
        # x.shape [1, 4, 240, 240]
        x = volume[:, :, :, i].unsqueeze(0).to(device)

        # Forward pass
        with autocast(device, dtype=next(model.parameters()).dtype):
            x_hat = model(x)

        # Only consider brain region for difference calculation
        mask_brain = (x > -2).float()
        diff = (x - x_hat) * mask_brain

        # Postprocessing
        diff_post = postprocessing(diff, x)

        # Collect results for each slice
        xs.append(x.detach().cpu())
        xs_hat.append(x_hat.detach().cpu())
        diffs.append(diff.detach().cpu())
        diffs_post.append(diff_post.detach().cpu())

    xs = torch.cat(xs, dim=0)
    xs_hat = torch.cat(xs_hat, dim=0)
    diffs = torch.cat(diffs, dim=0)
    diffs_post = torch.cat(diffs_post, dim=0)

    # Connect components
    ccs = get_connected_components(diffs_post)

    # Mask fusing
    mask_fused = mask_fusing(ccs)

    # Save results
    save_results(volume_key, mask_fused, output_path)
