import cv2
import numpy as np
import scipy.ndimage as ndi
from skimage import filters, morphology


def get_connected_components(diffs_post, diffs, intensity_component=None):
    diffs = np.asarray(diffs)
    diffs_post = np.transpose(np.asarray(diffs_post), (1, 0, 2, 3))  # Change to (modality, z, x, y) format

    # Initialize an empty array to store the largest components for each modality
    largest_components = np.zeros_like(diffs_post, dtype=np.uint8)

    # Iterate over each modality in diffs along axis 0
    for modality in range(diffs.shape[1]):
        diffs_post_modality = diffs_post[modality]
        diffs_modality = diffs[modality]
        # Label connected components for the current modality
        labeled_volume, num_features = ndi.label(diffs_post_modality)

        if num_features == 0:
            continue  # Skip if no features are found for this modality

        if intensity_component:
            intensities = np.zeros(num_features)

            for i in range(1, num_features + 1):
                component_voxels = diffs_modality[labeled_volume == i]

                if intensity_component == 'mean':
                    intensities[i - 1] = component_voxels.mean()
                elif intensity_component == 'sum':
                    intensities[i - 1] = component_voxels.sum()
                elif intensity_component == 'max':
                    intensities[i - 1] = component_voxels.max()
                else:
                    raise ValueError("intensity_component must be 'mean', 'sum', or 'max'.")

            largest_component_index = np.argmax(intensities) + 1  
        else:
            sizes = np.bincount(labeled_volume.ravel())
            sizes[0] = 0  # Ignore background
            largest_component_index = np.argmax(sizes)  

        largest_component = (labeled_volume == largest_component_index).astype(np.uint8)

        # Store the largest component for the current modality
        largest_components[modality] = largest_component

        # plot the largest component
        #import matplotlib.pyplot as plt
        #plt.imshow(largest_components[modality][5], cmap='gray')
        #plt.title(f'Diff largest component for modality {modality}')
        #plt.axis('off')
        #plt.show()
        #plt.close()
        
    return np.transpose(largest_components, (2, 3, 1, 0))  # Change back to (z, modality, x, y) format


def thresholding(diff, threshold):
    diff[diff < 0 ] = 0
    diff[diff > threshold] = 1
    return diff


def otsu(diff):
    # if diff contains nans, replace them with 0
    diff = np.nan_to_num(diff, nan=0.0)
    thr = filters.threshold_otsu(diff)
    diff[diff < thr] = 0
    diff[diff >= thr] = 1
    return diff


def brain_contour_f(diff, x):
    threshold = -0.95
    image_bin = np.zeros_like(x.cpu().numpy())
    image_bin[x.cpu().numpy() > threshold] = 1

    kernel = np.ones((10, 10), np.uint8)
    o_image = cv2.morphologyEx(image_bin, cv2.MORPH_OPEN, kernel)
    c_image = cv2.morphologyEx(image_bin, cv2.MORPH_CLOSE, kernel)

    o_contour = o_image - image_bin
    c_contour = c_image - image_bin

    o_contour_bin = np.zeros_like(o_contour)
    o_contour_bin[o_contour != 0] = 1

    c_contour_bin = np.zeros_like(c_contour)
    c_contour_bin[c_contour != 0] = 1

    sum_contour = o_contour_bin + c_contour_bin

    diff = diff - sum_contour
    diff[diff < 0] = 0
    return diff


def circle_masking_f(diff, circle_masking, convex_hull):
    final_c = np.squeeze(np.squeeze(diff, axis=0), axis=0).astype(np.uint8)
    contours, _ = cv2.findContours(final_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        largest_contour = max(contours, key=cv2.contourArea)

        if circle_masking:
            # Fit a minimum enclosing circle around the largest anomaly
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)
            center = (int(x), int(y))
            radius = int(radius)

            # Create a blank circular mask
            circular_mask = np.zeros_like(final_c)
            cv2.circle(circular_mask, center, radius, 255, thickness=-1)  # Fill the circle

            # Apply the circular mask to the original MRI
            diff = cv2.bitwise_and(final_c, final_c, mask=circular_mask)

        if convex_hull:
            hull = cv2.convexHull(largest_contour)
            convex_hull_mask = np.zeros_like(diff)
            diff = cv2.drawContours(convex_hull_mask, [hull], 0, 255, thickness=cv2.FILLED)

    if diff.ndim == 2:
        diff = np.expand_dims(np.expand_dims(diff, axis=0),axis=0)
    return diff


def postprocessing(diffs, x, postprocess, brain_contour, convex_hull, circle_masking):

    modalities = diffs.shape[0]
    diff_post = []

    for modality in range(modalities):
        if postprocess == 'StRegA':
            diff = thresholding(diffs[modality], threshold=0.2)

            diff = otsu(diff)                   

            diff = morphology.remove_small_objects(diff.astype(bool), min_size=64, connectivity=1).astype(int)

            if brain_contour:
                diff = brain_contour_f(diff, x)

            if circle_masking or convex_hull:
                diff = circle_masking_f(diff, circle_masking, convex_hull)
        
        elif postprocess == 'StRegA_t2f':
            diff = thresholding(diffs[modality], threshold=0.15)

            diff = otsu(diff)

            diff = morphology.remove_small_objects(diff.astype(bool), min_size=128, connectivity=1).astype(int)

            if brain_contour:
                diff = brain_contour_f(diff, x)

            if circle_masking:
                diff = circle_masking_f(diff, convex_hull)

        if postprocess == 'StRegA_multimodal':
            diff = thresholding(diffs[modality], threshold=0.3)

            diff = otsu(diff)                   

            diff = morphology.remove_small_objects(diff.astype(bool), min_size=64, connectivity=1).astype(int)

            if brain_contour:
                diff = brain_contour_f(diff, x)

            if circle_masking or convex_hull:
                diff = circle_masking_f(diff, circle_masking, convex_hull)

        elif postprocess == 'not_scaled' or postprocess == 'not_scaled_brain_mask':
            diff = diffs[modality]
            threshold = 0.2 * np.max(diff)

            diff[diff < threshold] = 0

            diff = otsu(diff)
            
            diff = morphology.remove_small_objects(diff.astype(bool), min_size=64, connectivity=1).astype(int)

            if brain_contour:
                diff = brain_contour_f(diff, x)
            
            if circle_masking:
                diff = circle_masking_f(diff, convex_hull)

        elif postprocess == 'not_scaled_multimodal':
            diff = diffs[modality]

            # plot the diff
            import matplotlib.pyplot as plt
            plt.imshow(diff, cmap='viridis')
            plt.title(f'Difference for modality {modality}')
            plt.colorbar()
            #plt.show()

            if modality == 1:
                threshold = 0.8
                diff[diff < threshold] = 0
            if modality == 2:
                threshold = 1.1
                diff[diff < threshold] = 0

            threshold = 0.2 * np.max(diff)

            diff[diff < threshold] = 0

            plt.imshow(diff, cmap='viridis')
            plt.title(f'Diff thesh for modality {modality}')
            plt.colorbar()
            #plt.show()

            diff = otsu(diff)

            plt.imshow(diff, cmap='gray')
            plt.title(f'Diff binarized for modality {modality}')
            #plt.show()

            diff = morphology.remove_small_objects(diff.astype(bool), min_size=64, connectivity=1).astype(int)

            plt.imshow(diff, cmap='gray')
            plt.title(f'Diff removed small components for modality {modality}')
            #plt.show()

            plt.close()

            if brain_contour:
                diff = brain_contour_f(diff, x)
            
            if circle_masking:
                diff = circle_masking_f(diff, convex_hull)

        elif postprocess == 'not_scaled_t1n':
            threshold = 0.8
            diff = diffs[modality]
            diff[diffs[modality] < threshold] = 0

            threshold = 0.2 * np.max(diff)
            diff[diff < threshold] = 0

            diff = otsu(diff)

            diff = morphology.remove_small_objects(diff.astype(bool), min_size=64, connectivity=1).astype(int)

            if brain_contour:
                diff = brain_contour_f(diff, x)
            
            if circle_masking:
                diff = circle_masking_f(diff, convex_hull)
        

        elif postprocess == 'not_scaled_t2f':
            threshold = 1.1
            diff = diffs[modality]
            diff[diffs[modality] < threshold] = 0

            threshold = 0.2 * np.max(diff)
            diff[diff < threshold] = 0

            diff = otsu(diff)

            diff = morphology.remove_small_objects(diff.astype(bool), min_size=64, connectivity=1).astype(int)

            if brain_contour:
                diff = brain_contour_f(diff, x)
            
            if circle_masking:
                diff = circle_masking_f(diff, convex_hull)
        
        elif postprocess == 'HistEq':
            brain = x.cpu().numpy()
            mask = np.zeros_like(brain)
            mask[brain > 0] = 1

            diff = cv2.equalizeHist((brain * 255).astype(np.uint8), mask * 255) * mask / 255

            diff = np.expand_dims(np.expand_dims(diff, axis=0), axis=0)

        elif postprocess == 'AdaptiveHistEq':
            brain = x.cpu().numpy()
            mask = np.zeros_like(brain)
            mask[brain > 0] = 1

            diff = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply((brain * 255).astype(np.uint8)) * mask / 255

            diff = np.expand_dims(np.expand_dims(diff, axis=0), axis=0)
        
        diff_post.append(diff)
    diffs_post = np.concatenate(np.expand_dims(diff_post, axis=0), axis=0)
    return diffs_post