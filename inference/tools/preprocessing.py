import os
import torch
import numpy as np
import nibabel as nib

def load_patient_volume(volume_key, input_path):
    modalidades = ["t1c", "t1n", "t2f", "t2w"]
    volume = []

    volume_path = os.path.join(input_path, volume_key)
    for mod in modalidades:
        volume_file = os.path.join(volume_path, f"{volume_key}-{mod}.nii.gz")
        img = nib.load(volume_file).get_fdata()
        img = (img - img.mean()) / img.std() 

        volume.append(img)

    volume_np = np.stack(volume, axis=0)  # shape [4, H, W, D]
    return torch.from_numpy(volume_np).float()
