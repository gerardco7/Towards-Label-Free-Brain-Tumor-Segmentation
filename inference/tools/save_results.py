import os
import nibabel as nib
import numpy as np


def save_results(volume_key, mask_fused, submission_path):
    # Ensure the submission path exists
    os.makedirs(submission_path, exist_ok=True)

    submission_img = nib.Nifti1Image(mask_fused, np.eye(4))
    nib.save(submission_img, os.path.join(submission_path, f"{volume_key}.nii.gz"))