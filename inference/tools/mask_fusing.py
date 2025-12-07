import numpy as np
from scipy.ndimage import binary_fill_holes


def fill_holes(mask):
    result = mask.copy()
    
    for region_value in [2, 3]:
        binary = (mask == region_value)
        
        filled = binary_fill_holes(binary)
        
        holes = np.logical_and(filled, ~binary)
        
        result[holes] = 1
    
    return result

def mask_fusing(ccs):
    """Fuses multiple segmentation masks to generate three types of brain tumor regions:
    NET (Necrotic and Non-Enhancing Tumor), EDEMA, and ET (Enhancing Tumor).
    Args:
        ccs (dict): Dictionary containing connected component masks for T2f and T1c modalities.
        sam_mask (dict): Dictionary containing segmentation masks from the SAM model for T2f and T1c modalities.
        plot (bool): Whether to display plots of the fused masks.
        save_path (str): Path to save the resulting fused masks.
        volume_key (str): Identifier for the current volume being processed.
    Returns:
        dict: Dictionary with fused masks for NET, EDEMA, and ET regions.
    Notes:
        - ET is taken directly from the T1c mask in sam_mask.
        - EDEMA is computed as the T2f mask in sam_mask minus the T1c mask in sam_mask.
        - NET is computed as the regions in ccs masks that are not covered by T2f and T1c.
    """
    mask_fused = np.zeros_like(ccs[0], dtype=np.uint8)
    
    t1c_mask = ccs[0, :, :, :]
    t2f_mask = ccs[2, :, :, :]

    mask_fused[t2f_mask > 0] = 2 # EDEMA
    mask_fused[t1c_mask > 0] = 3 # ET

    mask_fused = fill_holes(mask_fused)  


    return mask_fused



