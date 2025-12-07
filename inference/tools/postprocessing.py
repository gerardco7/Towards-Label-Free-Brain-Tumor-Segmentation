import torch
import scipy.ndimage as ndimage
import cv2
from skimage import filters
from skimage.morphology import opening, disk

import matplotlib.pyplot as plt
import numpy as np


def postprocessing(diff, x,):
    # x.shape [1, 4, 240, 240]
    device = diff.device
    batch, modalities = x.shape[:2]

    thresholds = [1.2, 0.8, 1.5, 0.8]
    r_thresholds = [0.2, 0.2, 0.2, 0.2]

    diff_post_1 = torch.zeros_like(diff)
    diff_post_2 = torch.zeros_like(diff)
    diff_post_3 = torch.zeros_like(diff)
    diff_post_4 = torch.zeros_like(diff)

    for modality in range(modalities):
        if modality == 0 or modality == 2:
            # Step 1a: Absolute threshold
            threshold = thresholds[modality]
            diff_post_1[:, modality] = torch.where(
                diff[:, modality] > threshold, diff[:, modality], torch.tensor(0.0, device=device)
            )

            # Step 1b: Relative threshold
            r_threshold = r_thresholds[modality]
            diff_post_2[:, modality] = torch.where(
                diff_post_1[:, modality] > r_threshold, diff_post_1[:, modality], torch.tensor(0.0, device=device)
            )

            # Step 2: Otsu thresholding
            arr = diff_post_2[:, modality].detach().cpu().numpy()
            otsu_threshold = filters.threshold_otsu(arr)
            diff_post_3[:, modality] = torch.where(
                diff_post_2[:, modality] > otsu_threshold,
                torch.tensor(1.0, device=device),
                torch.tensor(0.0, device=device)
            )
            # Step 3: Morphological opening and closing
            selem = disk(1)
            opened = opening(diff_post_3[0, modality].detach().cpu().numpy(), selem)
            closed = ndimage.binary_closing(opened, structure=selem)
            diff_post_4[:, modality] = torch.tensor(closed, device=device).unsqueeze(0)

    return diff_post_4


def get_connected_components(diffs_post):
    # diffs_post: [155, 4, 240, 240]

    diffs_cc = np.array(diffs_post.detach().cpu().numpy())


    ccs = np.transpose(np.zeros_like(diffs_cc, dtype=np.uint8), (1, 0, 2, 3))  # [155, 4, 240, 240] to [4, 155, 240, 240]

    for modality in range(diffs_cc.shape[1]):
        labeled_volume, num_features = ndimage.label(diffs_cc[:, modality, :, :])

        sizes = np.bincount(labeled_volume.ravel())
        sizes[0] = 0
        largest_label_index = sizes.argmax()
    
        cc = (labeled_volume == largest_label_index).astype(np.uint8)

        ccs[modality] = cc

    return np.transpose(np.array(ccs), (0, 2, 3, 1))  # [4, 155, 240, 240] to [4, 240, 240, 155]


def sam_masking(xs, ccs, predictor):
    # xs: [155, 4, 240, 240]
    # ccs: [4, 240, 240, 155]
    xs_np = xs.detach().cpu().numpy()
    sam_masks = np.zeros_like(xs_np, dtype=np.uint8)  # [155, 4, 240, 240]

    for i in range(xs_np.shape[0]):
        for modality in range(xs_np.shape[1]):
            if modality not in [0, 2]:
                continue

            cc_mask = ccs[modality, :, :, i]
            if cc_mask.sum() == 0:
                sam_masks[i, modality] = cc_mask
                continue

            img = cv2.normalize(xs_np[i, modality], None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
            predictor.set_image(img_rgb)

            x0, y0, w, h = cv2.boundingRect(cc_mask.astype(np.uint8))
            x1, y1 = x0 + w, y0 + h
            input_box = np.array([x0, y0, x1, y1])

            points = np.column_stack(np.where(cc_mask > 0))
            if len(points) > 5:
                idx = np.random.choice(len(points), 5, replace=False)
                input_point = points[idx][:, [1, 0]]  # swap x/y
            else:
                input_point = points[:, [1, 0]]

            input_label = np.ones(len(input_point), dtype=np.int32)
            mask_found = False

            for it in range(3):
                if len(input_point) == 0:
                    break
                masks, scores, _ = predictor.predict(
                    box=input_box,
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True,
                )
                if scores[0] > 0.9:
                    sam_masks[i, modality] = masks[0].astype(np.uint8)
                    mask_found = True
                    break

            if not mask_found:
                sam_masks[i, modality] = cc_mask.astype(np.uint8)

    return np.transpose(sam_masks, (1, 2, 3, 0))  # [155, 4, 240, 240] to [4, 240, 240, 155]
