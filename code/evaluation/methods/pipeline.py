import os
import time
import pandas as pd
import torch
import numpy as np
import cv2
from torch.amp import autocast
from torchvision import transforms
import matplotlib.pyplot as plt

from methods.metrics import compute_metrics, combine_metrics
from methods.postprocessing import postprocessing, get_connected_components
from methods.plot import plot_results
from methods.read_data import read_data
from methods.ensamble import ensamble_method
from methods.simpletik_reader_writer import SimpleITKIO

import sys

sys.path.append('../models')

from get_network_from_plans import get_patch_size_from_plans


def preprocess_volume(volume, masks, patch_size):
    """Resize volume and masks to match the patch size."""  
    # 1 182 218 X -> X 1 240 240
    volume = volume.permute(3, 0, 1, 2) 
    volume = torch.nn.functional.interpolate(
        volume, size=patch_size, mode='bilinear', align_corners=False
    ).permute(1, 2, 3, 0)

    masks = masks.permute(2, 0, 1)
    masks = torch.nn.functional.interpolate(
        masks.unsqueeze(0), size=patch_size, mode='bilinear', align_corners=False
    ).squeeze(0).permute(1, 2, 0)
    return volume, masks


def apply_transformation(x, transform_type):
    """Apply the specified transformation to the input tensor."""
    if transform_type == 'minmax[-1,1]':
        transformation = transforms.Compose([
            transforms.Lambda(lambda x: torch.stack([(x[:, i] - x[:, i].min()) / (x[:, i].max() - x[:, i].min()) * 2 - 1 for i in range(x.shape[1])], dim=1))
        ])
    elif transform_type == 'base':
        transformation = transforms.Compose([
            transforms.Lambda(lambda x: x)
        ])

    trans = transformation(x)

    return trans


def process_slice(model, x, postprocess, nnUnet, device):
    """Process a single slice through the model."""
    if postprocess not in ['HistEq', 'AdaptiveHistEq']:
        if postprocess == 'not_scaled_brain_mask':
            return (x > -2).float(), None
        else:
            with autocast(device, dtype=next(model.parameters()).dtype):
                if nnUnet:
                    return model(x).sum(axis=1).squeeze(0).cpu().detach().numpy()
                else:
                    return model(x)
                
def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    
def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)   
    
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=2))   

                
def SAM_masking(x, diff_posts, predictor, SAM_ensemble):
    """Apply SAM masking to the input tensor."""

    stacked_masks = []
    plot = False

    for i in range(len(x)):
        diff_post = diff_posts[ :, :, i, 0]

        img = cv2.normalize(x[i].squeeze(0), None, 0, 255, cv2.NORM_MINMAX)
        img = img.astype(np.uint8)

        # Convert grayscale to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        predictor.set_image(img_rgb)

        scores = [0, 0, 0]
        it = 0

        # If diff post is empty, skip
        if diff_post.sum() == 0:
            it = 10

        else:
            # obtain the bounding box of the diff_post
            x0, y0, w, h = cv2.boundingRect(diff_post.astype(np.uint8))
            x1 = x0 + w
            y1 = y0 + h
            input_box = np.array([x0, y0, x1, y1])

        # Select where diff_post > 0 and transpose dims 1 and 0
        points = np.array(np.where(diff_post > 0)).T[:, [1, 0]]

        while scores[0] < 0.9 and it < 3:
            if len(points) >= 5:
                idx = np.random.choice(len(points), 5, replace=False)
                input_point = points[idx]
            else:
                input_point = points

            input_label = np.array([1] * len(input_point))

            masks, scores, logits = predictor.predict(
                box=input_box,
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True,
            )   

            if plot:
                # plot mask with diff_post
                for j, (mask, score) in enumerate(zip(masks, scores)):
                    if j == 0:
                        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
                        axs[0].imshow(mask, cmap='gray')
                        #axs[0].imshow(img_rgb)
                        #show_mask(mask, axs[0])
                        #show_box(input_box, axs[0])
                        #show_points(input_point, input_label, axs[0])
                        axs[0].set_title(f"Mask {j+1}, Score: {score:.3f}", fontsize=18)
                        axs[0].axis('off')
                        axs[1].imshow(diff_post, cmap='gray')
                        axs[1].set_title("diff_post", fontsize=18)
                        axs[1].axis('off')
                        plt.tight_layout()
                        plt.show()
            it += 1
        
        # If the score is high, use the first mask
        if scores[0] > 0.9:
            if SAM_ensemble:
                ensembled_mask = ensamble_method([masks[0], diff_post], SAM_ensemble)
                stacked_masks.append(ensembled_mask)

                # if plot show mask[0], diffs post and the ensembled mask
                if plot:
                    fig, axs = plt.subplots(1, 3, figsize=(12, 6))
                    axs[0].imshow(masks[0])
                    axs[0].set_title(f"Mask SAM", fontsize=18)
                    axs[0].axis('off')
                    axs[1].imshow(diff_post, cmap='gray')
                    axs[1].set_title("diff_post", fontsize=18)
                    axs[1].axis('off')
                    axs[2].imshow(ensembled_mask, cmap='gray')
                    axs[2].set_title("Ensembled Mask", fontsize=18)
                    axs[2].axis('off')
                    plt.tight_layout()
                    plt.show()


            else:
                stacked_masks.append(masks[0])
        else:
            # If the score is low, use the original diff_post
            stacked_masks.append(diff_post)


        
    return np.expand_dims(np.stack(stacked_masks, axis=-1), axis=-1)


def handle_connected_components(xs, xs_hat, diffs, diffs_post, masks, metrics, plot, data, volume_key, ensamble=False, t1c_diffs=None):
    """Handle connected components and compute metrics."""
    # If diffs post is x, 1, 240, 240, then we need to transpose it to 240, 240, x, 1
    if diffs_post.shape[1] == 1:
        diffs_post = diffs_post.transpose(2, 3, 0, 1)
    cc = torch.stack([torch.tensor(c) for c in diffs_post], dim=0)
    diffs = torch.stack([torch.tensor(diff) for diff in diffs], dim=-1)
    xs = torch.stack([torch.tensor(x) for x in xs], dim=-1)
    xs_hat = torch.stack([torch.tensor(x_hat) for x_hat in xs_hat], dim=-1)

    for i in range(xs.shape[-1]):
        x = xs[:, :, :, i]
        x_hat = xs_hat[:, :, :, i]
        diff = diffs[:, :, :, i]
        if cc.ndim == 4:
            diff_post = cc[:, :, i, 0]
        else:
            diff_post = cc[:, :, i]
        mask = masks[:, :, i]
        if t1c_diffs is not None:
            t1c_diff = t1c_diffs[:, :, i]
        else:
            t1c_diff = None

        if metrics:
            compute_metrics(data, x, x_hat, mask, diff_post, metrics, volume_key, i, ensamble, t1c_diff)
        if plot:
            plot_results(x, x_hat, diff, diff_post, mask, data[-len(metrics):])


def pipeline(modalities, test_modalities, models, datasets, data_keys, transform, device, plot, postprocess, brain_contour, convex_hull,
             circle_masking, connected_components, intensity_component, metrics, ensamble, multimodal, bimodal, nnUnet, SAM, SAM_ensemble, plans_path):
    """
    Pipeline for evaluation.

    Parameters
    ----------
    modalities : list
        List of modalities.
    test_modalities : list
        List of test modalities.
    models : list
        List of models.
    datasets : list
        List of datasets.
    data_keys : list
        List of data keys.
    transform : list
        List of transformations.
    device : str
        Device to use.
    plot : bool
        Whether to plot results.
    postprocess : list
        List of postprocessing methods.
    brain_contour : bool
        Whether to use brain contour.
    convex_hull : bool
        Whether to use convex hull.
    circle_masking : bool
        Whether to use circle masking.
    connected_components : bool
        Whether to use connected components.
    intensity_component : bool
        Whether to use intensity component.
    metrics : list
        List of metrics to compute.
    ensamble : bool
        Whether to use ensemble method.
    multimodal : bool
        Whether to use multimodal processing.
    bimodal : bool
        Whether to use bimodal processing.
    nnUnet : bool
        Whether to use nnUnet.
    plans_path : str
        Path to nnUnet plans.

    Returns
    -------
    pd.DataFrame
        Results dataframe.
    """
    data = []
    patch_size = [240, 240]

    index_i = 0

    if SAM:
        sys.path.append('C:/Users/gerar/Documents/TFG/segment-anything')
        from segment_anything import SamPredictor, SamAutomaticMaskGenerator, sam_model_registry

        sam = sam_model_registry["vit_h"](checkpoint="D:/models/SAM/sam_vit_h_4b8939.pth")
        sam.to(device="cuda:0")
        predictor = SamPredictor(sam)
    else:
        predictor = None

    for volume_key in data_keys:
        start = time.time()
        diffs_post_all = []

        index_i += 1

        for j, model in enumerate(models):
            model.eval()

            if nnUnet:
                nnUnet_raw = r"D:\data\nnUnet_raw"
                volume, masks = [], []

                for i in range(len(modalities[0])):
                    img, _ = SimpleITKIO().read_images(
                        [os.path.join(nnUnet_raw, f'Dataset001_BraTSMEN/imagesTr/BraTS-MEN-00004_000{i}.nii.gz')]
                    )
                    volume.append(img)

                label, _ = SimpleITKIO().read_images(
                    [os.path.join(nnUnet_raw, 'Dataset001_BraTSMEN/labelsTr/BraTS-MEN-00004.nii.gz')]
                )
                volume = torch.tensor(np.array(volume), dtype=torch.float32).squeeze(1)
                masks = torch.tensor(label, dtype=torch.float32).squeeze(0)
                patch_size = get_patch_size_from_plans(plans_path)
            else:
                volume, masks = read_data(datasets[j], volume_key, modalities[j])

            if volume.shape[1:3] != tuple(patch_size):
                volume, masks = preprocess_volume(volume, masks, patch_size)

            xs, xs_hat, diffs, diffs_post = [], [], [], []

            for i in range(volume.shape[-1]):
                x = volume[:, :, :, i].unsqueeze(0).to(device).to(next(model.parameters()).dtype)
                x = apply_transformation(x, transform[j])
                mask = masks[:, :, i]

                x_hat, _, _, _ = process_slice(model, x, postprocess[j], nnUnet, device)

                x_hat = x_hat.squeeze(0)[test_modalities[j]].cpu().detach().numpy()
                x = x.squeeze(0)[test_modalities[j]].cpu().detach().numpy()

                if not nnUnet:
                    mask_brain = (x > -2).astype(float)
                    diff = (x - x_hat) * mask_brain

                    if postprocess[j]:
                        diff_post = postprocessing(
                            diff, x, postprocess[j],
                            brain_contour, convex_hull, circle_masking
                        )
                if connected_components or ensamble:
                    xs.append(x)
                    xs_hat.append(x_hat)
                    diffs_post.append(diff_post)
                    diffs.append(diff)
                else:

                    if metrics:
                        compute_metrics(data, x, x_hat, mask, diff_post, metrics, volume_key, i)
                    if plot:
                        plot_results(x, x_hat, diff, diff_post, mask, data[-len(metrics):])

            if connected_components:
                diff_post = get_connected_components(diffs_post, diffs, intensity_component=intensity_component)
                if SAM: 
                    diff_post = SAM_masking(xs, diff_post, predictor, SAM_ensemble)

                if ensamble or multimodal:
                    diffs_post_all.append(diff_post)
                else:
                    handle_connected_components(xs, xs_hat, diffs, diff_post, masks, metrics, plot, data, volume_key)

        if ensamble or multimodal:
            if multimodal:
                ensamble = "sum"
                # [1, 240, 240, 20, 4] -> [4, 240, 240, 20]
                diffs_post_all = np.array(diffs_post_all[0]).transpose(3, 0, 1, 2)
            else:
                # [x, 240, 240, 20, 1] -> [x, 240, 240, 20]
                # [x, 1, 240, 240, 20] -> [x, 240, 240, 20]
                diffs_post_all = np.array(diffs_post_all).squeeze(-1)
            diffs_post = ensamble_method(diffs_post_all, ensamble)

            if bimodal:
                t1c_diffs = diffs_post_all[0]
            else:
                t1c_diffs = None
            handle_connected_components(xs, xs_hat, diffs, diffs_post, masks, metrics, plot, data, volume_key, ensamble, t1c_diffs)
        end = time.time()
        print(f'Volume: {volume_key}, Time: {end - start}')

    df = pd.DataFrame(data, columns=['Volume', 'Slice', 'Metric', 'Value'])
    return combine_metrics(df)
