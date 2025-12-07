import numpy as np
import pandas as pd
import csv
import torch 

import os
import sys

from scipy.spatial.distance import directed_hausdorff

from methods.plot import create_results_plot

sys.path.append(os.path.abspath('../methods/losses'))
from losses import ssim_loss


def mse(true_mask, pred_mask):
    assert true_mask.shape == pred_mask.shape
    true_mask = np.asarray(true_mask, dtype=np.float32)
    pred_mask = np.asarray(pred_mask, dtype=np.float32)

    mse_value = np.mean((true_mask - pred_mask) ** 2)
    return mse_value


def psnr(true_mask, pred_mask):
    assert true_mask.shape == pred_mask.shape
    true_mask = np.asarray(true_mask, dtype=np.float32)
    pred_mask = np.asarray(pred_mask, dtype=np.float32)

    mse_value = mse(true_mask, pred_mask)
    if mse_value == 0:
        return float('inf')
    
    psnr_value = 20 * np.log10(255.0 / np.sqrt(mse_value))
    return psnr_value


def ssim(true_mask, pred_mask):
    assert true_mask.shape == pred_mask.shape

    true_mask = true_mask.to(dtype=torch.float32).unsqueeze(0)
    pred_mask = pred_mask.to(dtype=torch.float32).unsqueeze(0)
    
    ssim_value = ssim_loss(true_mask, pred_mask)
    return ssim_value
    

def dice(true_mask, pred_mask, non_seg_score=1.0, ensamble=False, t1c_diff=None, t2f_diff=None):
    """
    Compute the Dice coefficient for ET, TC, and WT regions.

    Parameters:
    - true_mask: Ground truth mask (numpy array), shape: {true_mask.shape if 'true_mask' in locals() else 'unknown'}
    - pred_mask: Predicted mask (numpy array), shape: {pred_mask.shape if 'pred_mask' in locals() else 'unknown'}
    - non_seg_score: Score to assign when no segmentation is present.
    - ensamble: Boolean indicating whether to use ensemble predictions.
    - t1c_diff: Optional mask for ensemble predictions, shape: {t1c_diff.shape if t1c_diff is not None else 'None'}
    - t2f_diff: Optional mask for ensemble predictions, shape: {t2f_diff.shape if t2f_diff is not None else 'None'}

    Returns:
    - List of Dice coefficients for ET, TC, WT, and Edema regions.
    """
    assert true_mask.shape == pred_mask.shape, f"Shape mismatch between true_mask {true_mask.shape} and pred_mask {pred_mask.shape}"
    true_mask = np.asarray(true_mask, dtype=np.int32)
    pred_mask_c = np.asarray(pred_mask, dtype=np.bool_)
    t1c_mask = np.asarray(t1c_diff, dtype=np.bool_) if t1c_diff is not None else None
    t2f_mask = np.asarray(t2f_diff, dtype=np.bool_) if t2f_diff is not None else None

    # Define regions
    regions = {
        "ET": true_mask == 3,          # Enhancing Tumor
        "TC": np.logical_or(true_mask == 1, true_mask == 3),  # Tumor Core (1 + 3)
        "WT": true_mask >= 1,          # Whole Tumor (1 + 2 + 3)
        "Edema": true_mask == 2,       # Edema (2)
    }

    dice_scores = []


    for region_name, region_mask in regions.items():
        if t1c_diff is not None and region_name in ["ET", "TC"]:
            pred_mask_c = t1c_mask
        elif t2f_diff is not None and region_name in ["Edema"]:
            pred_mask_c = t2f_mask
        else:
            pred_mask_c = np.asarray(pred_mask, dtype=np.bool_)

        intersection = np.logical_and(region_mask, pred_mask_c).sum()
        union = region_mask.sum() + pred_mask_c.sum()

        if union == 0:
            dice_score = non_seg_score
        else:
            dice_score = 2.0 * intersection / union

        dice_scores.append(dice_score)
    return dice_scores


def hausdorff_distance(true_mask, pred_mask):
    assert true_mask.shape == pred_mask.shape
    true_mask = np.asarray(true_mask, dtype=np.bool_)
    pred_mask = np.asarray(pred_mask, dtype=np.bool_)

    true_points = np.argwhere(true_mask)
    pred_points = np.argwhere(pred_mask)

    if len(true_points) == 0 or len(pred_points) == 0:
        return np.inf

    forward_hd = directed_hausdorff(true_points, pred_points)[0]
    backward_hd = directed_hausdorff(pred_points, true_points)[0]

    return max(forward_hd, backward_hd)


def confusion_matrix(true_mask, pred_mask, threshold = 0.5):
    assert true_mask.shape == pred_mask.shape
    true_mask = np.asarray(true_mask, dtype=np.bool_)
    pred_mask = np.asarray(pred_mask, dtype=np.bool_)

    true_positive = np.logical_and(true_mask, pred_mask).sum() 
    true_negative = np.logical_and(np.logical_not(true_mask), np.logical_not(pred_mask)).sum()
    false_positive = np.logical_and(np.logical_not(true_mask), pred_mask).sum()
    false_negative = np.logical_and(true_mask, np.logical_not(pred_mask)).sum()

    return [true_positive, true_negative, false_positive, false_negative]


def compute_metrics(data, x, x_hat, true_mask, pred_mask, metrics, volume, slice, ensamble=False, t1c_diff=None, t2f_diff=None):
    for metric in metrics:
        if metric == 'dice':
            value = dice(true_mask, pred_mask, ensamble=ensamble, t1c_diff=t1c_diff, t2f_diff=t2f_diff)
        elif metric == 'hausdorff_distance':
            value = hausdorff_distance(true_mask, pred_mask)
        elif metric == 'confusion_matrix':
            value = confusion_matrix(true_mask, pred_mask)
        elif metric == 'mse':
            value = mse(x, x_hat)
        elif metric == 'psnr':
            value = psnr(x, x_hat)
        elif metric == 'ssim':
            value = ssim(x, x_hat)
        else:
            raise ValueError(f"Unknown metric {metric}")
        data.append({"Volume": volume, "Slice": slice, "Metric": metric, "Value": value})

    return data


def combine_metrics(df):
    metrics_values = {}

    create_results_plot(df)

    for metric in df['Metric'].unique():
        metric_values = df[df['Metric'] == metric]['Value']
        if metric == 'confusion_matrix':
            metric_values = np.array(metric_values.tolist())
            metrics_values[metric] = {
                'true_positive': metric_values[:, 0].sum(),
                'true_negative': metric_values[:, 1].sum(),
                'false_positive': metric_values[:, 2].sum(),
                'false_negative': metric_values[: ,3].sum()
            }

            tp = metrics_values[metric]['true_positive']
            tn = metrics_values[metric]['true_negative']
            fp = metrics_values[metric]['false_positive']   
            fn = metrics_values[metric]['false_negative']

            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            metrics_values[metric]['recall'] = recall
            metrics_values[metric]['precision'] = precision
            metrics_values[metric]['f1_score'] = f1_score
            metrics_values[metric]['tp_percentage'] = tp / (tp + tn + fp + fn) * 100
            metrics_values[metric]['tn_percentage'] = tn / (tp + tn + fp + fn) * 100
            metrics_values[metric]['fp_percentage'] = fp / (tp + tn + fp + fn) * 100
            metrics_values[metric]['fn_percentage'] = fn / (tp + tn + fp + fn) * 100

        elif metric == 'dice':
            dice_values = np.array(metric_values.tolist())
            metrics_values[metric] = {
                'ET_avg': dice_values[:, 0].mean(),
                'ET_std': dice_values[:, 0].std(),
                'TC_avg': dice_values[:, 1].mean(),
                'TC_std': dice_values[:, 1].std(),
                'WT_avg': dice_values[:, 2].mean(),
                'WT_std': dice_values[:, 2].std(),
                'Edema_avg': dice_values[:, 3].mean(),
                'Edema_std': dice_values[:, 3].std()
            }
        elif metric == 'hausdorff_distance':
            metrics_values[metric] = np.percentile(metric_values.astype(float), 95)
        
        elif metric == 'mse':
            metrics_values[metric] = np.mean(metric_values.astype(float))
        
        elif metric == 'psnr':
            metrics_values[metric] = np.mean(metric_values.astype(float))

        elif metric == 'ssim':
            metrics_values[metric] = np.mean(metric_values.astype(float))
        else:
            raise ValueError(f"Unknown metric {metric}")

    print(metrics_values)
    return metrics_values   

def save_results(results, file_name):
    with open(file_name, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Metric', 'Value'])

        for key, value in results.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    writer.writerow([f'{key}_{sub_key}', sub_value])
            else:
                writer.writerow([key, value])