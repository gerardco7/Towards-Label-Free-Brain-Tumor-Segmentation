import os
import numpy as np
import nibabel as nib
import h5py
import argparse
import time
from tqdm import tqdm

from DatasetBlosc2 import DatasetBlosc2

"""
python3 adapt_BraTS4trainning.py --blosc2 True --path_out "D:/data/BraTS2023-MEN-adaped-Blosc2"
"""
def main(args):
    path_brats = args.path_brats
    path_out = args.path_out
    normalization = args.normalization
    _blosc2 = args.blosc2

    start = time.time()

    modalities = ['t1c', 't1n', 't2f', 't2w', 'seg']

    os.makedirs(path_out, exist_ok=True)
    os.makedirs(os.path.join(path_out, 'test'), exist_ok=True)
    os.makedirs(os.path.join(path_out, 'train'), exist_ok=True)

    if not _blosc2:
        for modality in modalities:
            os.makedirs(os.path.join(path_out, 'test', modality), exist_ok=True)
            os.makedirs(os.path.join(path_out, 'train', modality), exist_ok=True)

    patients = sorted(os.listdir(path_brats))

    for patient in tqdm(patients, desc="Processing patients"):
        path_patient = os.path.join(path_brats, patient)
        datas = [nib.load(os.path.join(path_patient, f"{patient}-{modality}.nii.gz")).get_fdata() for modality in modalities]
        
        if normalization == 'zscore':
            datas[:-1] = [(data - np.mean(data)) / np.std(data) for data in datas[:-1]]
        elif normalization == 'zscore_brain':
            datas[:-1] = [(data - np.mean(data[data > 0])) / np.std(data[data > 0]) for data in datas[:-1]]
        elif normalization == 'minmax':
            datas[:-1] = [(data - np.min(data)) / (np.max(data) - np.min(data)) for data in datas[:-1]]

        seg_data = datas[-1]

        if _blosc2:
            test_slices = [i for i in range(seg_data.shape[-1]) if np.any(seg_data[:, :, i] != 0)]
            train_slices = [i for i in range(seg_data.shape[-1]) if np.all(seg_data[:, :, i] == 0)]
        
            datas_array = np.array(datas[:-1]).astype(np.float32)
            test_volume = datas_array[:, :, :, test_slices]
            train_volume = datas_array[:, :, :, train_slices]

            seg_volume = seg_data[:, :, test_slices]

            block_size_test_data, chunk_size_test_data  = DatasetBlosc2.comp_blosc2_params(test_volume.shape, test_volume.shape[1:])
            block_size_train_data, chunk_size_train_data  = DatasetBlosc2.comp_blosc2_params(train_volume.shape, train_volume.shape[1:])
            block_size_seg, chunk_size_seg = DatasetBlosc2.comp_blosc2_params(seg_volume.shape, seg_volume.shape[1:])

            DatasetBlosc2.save_case(test_volume, seg_volume, os.path.join(path_out, 'test', patient), chunk_size_test_data, block_size_test_data, chunk_size_seg, block_size_seg, )
            DatasetBlosc2.save_case(train_volume, None, os.path.join(path_out, 'train', patient), chunk_size_train_data, block_size_train_data)

        else: 
            for slice_idx in range(seg_data.shape[2]):
                target_set = 'test' if np.sum(seg_data[:, :, slice_idx]) > 0 else 'train'
                for i, modality in enumerate(modalities):
                    file_name = f"{patient}_{slice_idx}_{modality}.h5"
                    path_out_set = os.path.join(path_out, target_set, modality, file_name)
                    with h5py.File(path_out_set, 'a') as hf:
                        hf.create_dataset('data', data=datas[i][:, :, slice_idx])

    print(f"Time: {time.time() - start}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adapt BraTS dataset')
    parser.add_argument('--path_brats', default='D:/data/BraTS2023-MEN', type=str)
    parser.add_argument('--path_out', default='D:/data/BraTS2023-MEN-adapted2', type=str)
    parser.add_argument('--normalization', default='zscore', type=str)
    parser.add_argument('--blosc2', default=False, type=bool)   
    args = parser.parse_args()
    main(args)
