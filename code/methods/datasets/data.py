import os
import torch
import h5py
import json
import time
import numpy as np
from tqdm import tqdm
import torchio as tio
from torch.utils.data import Dataset, DataLoader, TensorDataset
from torchvision import transforms
from typing import List, Tuple, Optional
from sklearn.model_selection import KFold

from methods.datasets.DatasetBlosc2 import DatasetBlosc2

def generate_crossval_split(train_identifiers: List[str], seed=12345, n_splits=5) -> List[dict[str, List[str]]]:
    splits = []
    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for i, (train_idx, test_idx) in enumerate(kfold.split(train_identifiers)):
        train_keys = np.array(train_identifiers)[train_idx]
        test_keys = np.array(train_identifiers)[test_idx]
        splits.append({})
        splits[-1]['train'] = list(train_keys)
        splits[-1]['val'] = list(test_keys)
    return splits
    

class Data(Dataset):
  def __init__(self, files: List[str], mode: str, base_path: str, modalities: List[str] = ['t1c', 't1n', 't2f', 't2w'], transform: Optional[transforms.Compose] = None):
    self.files = files
    self.mode = mode
    self.transform = transform
    self.modalities = modalities
    self.base_path = base_path

  def __getitem__(self, idx: int) -> torch.Tensor:
    while True:
      images = []
      for modality in self.modalities:
        path = os.path.join(self.base_path, modality, self.files[idx])
        with h5py.File(path, 'r') as f:
          data = f['data'][:]
        if data.max() == data.min():
          idx = (idx + 1) % len(self.files)
          break
        image = torch.tensor(data, dtype=torch.float32)
        if torch.isnan(image).any() or torch.isinf(image).any():
          print('NaN or Inf found in image', self.files[idx])
          idx = (idx + 1) % len(self.files)
          break
        if self.transform:
          image = self.transform(image)
        images.append(image)
        return torch.cat(images, dim=0)

  def __len__(self) -> int:
    return len(self.files)
  

class Blosc2Data(Dataset):
  def __init__(self, files: List[str], mode: str, base_path: str, modalities: List[str] = ['t1c', 't1n', 't2f', 't2w'], transform: Optional[transforms.Compose] = None, tio_bool: bool = False, dataset: DatasetBlosc2 = None):
    self.files = files
    self.mode = mode
    self.transform = transform
    self.tio_bool = tio_bool
    self.modalities = modalities
    self.base_path = base_path
    self.dataset = dataset
    self.modality_map = {
      't1c': 0,
      't1n': 1,
      't2f': 2,
      't2w': 3
    }
    self.mapped_modalities =None

  def __getitem__(self, idx: int) -> torch.Tensor:
    if self.mapped_modalities is None:
      self.mapped = [self.modality_map[modality] for modality in self.modalities]

    data, seg = self.dataset.load_case(self.files[idx])
    data = torch.tensor(np.array(data), dtype=torch.float32)
    data = data[self.mapped, ...] 


    if self.tio_bool:
      subject = tio.Subject(
          image=tio.ScalarImage(tensor=data)
      )
       
      data = self.transform(subject).image.data.unsqueeze(0)
    elif self.transform:
      data = self.transform(data)

    if seg is not None:
      seg = torch.tensor(np.array(seg), dtype=torch.float32)
      seg = seg.unsqueeze(0)
      return data.squeeze(0), seg
    
    return data
  
  def __len__(self) -> int:
    return len(self.files)

class SliceDataset(torch.utils.data.IterableDataset):
    def __init__(self, volume_dataset, mode='train'):
        self.volume_dataset = volume_dataset
        self.mode = mode
        self._length = 92274 # Placeholder for length, previously computed in __len__

    def __iter__(self):
        for volume in self.volume_dataset:
            if isinstance(volume, tuple):  # (data, seg)
                data, seg = volume
                for j in range(data.shape[-1]):
                    yield data[..., j], seg[..., j]
            else:
                for j in range(volume.shape[-1]):
                    yield volume[..., j], None

    def __len__(self):
        if self._length is not None:
            return self._length

        # Only compute this once
        print("Precomputing dataset length (may take a while)...")
        length = 0
        for volume in tqdm(self.volume_dataset, desc=f"Calculating {self.mode} dataset length"):
            if isinstance(volume, tuple):
                data = volume[0]
            else:
                data = volume
            length += data.shape[-1]
        self._length = length
        print("Dataset length computed.", length)
        return length


def create_dataloader(dataset, batch_size, shuffle):
    """
    Create a DataLoader with a given dataset.
    """
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, 
                         collate_fn=lambda x: torch.stack([item[0] for item in x]))


def create_dataset(files, mode, data_path, modalities, transform, tio_bool, dataset=None, blosc2=False):
    """
    Create the appropriate dataset class based on the `blosc2` flag.
    """
    if blosc2:
        return Blosc2Data(files, mode, base_path=data_path, modalities=modalities, transform=transform, tio_bool=tio_bool, dataset=dataset)
    else:
        return Data(files, mode, base_path=data_path, modalities=modalities, transform=transform)


def loadData(data_path, modalities, transform, batch_size, blosc2=False, fold=0, test=False):
    """
    Load data for training, validation, or testing, and return corresponding DataLoaders.
    """
    # Load splits if blosc2 is used
    if blosc2:
        dataset = DatasetBlosc2(folder=data_path, identifiers=None)
        
        if not test:
            splits_file = os.path.join(data_path, "splits_final.json")
            if not os.path.isfile(splits_file):
                all_keys_sorted = list(np.sort(list(dataset.identifiers)))
                print(f"Total number of patients: {len(all_keys_sorted)}")
                print(f"data_path: {data_path}")
                splits = generate_crossval_split(all_keys_sorted, seed=12345, n_splits=5)
                with open(splits_file, 'w') as f:
                    json.dump(splits, f)
                print("Splits file saved")
            else: 
                with open(splits_file, 'r') as f:
                    splits = json.load(f)

            train_files = splits[fold]['train']
            valid_files = splits[fold]['val']
        else:
            test_files = dataset.get_identifiers(data_path)
            print(f"Found {len(test_files)} cases in folder {data_path}")

    else:
        image_files = os.listdir(os.path.join(data_path, 't1c'))
        if not test:
            split_idx = int(0.8 * len(image_files))
            train_files, valid_files = image_files[:split_idx], image_files[split_idx:]
        else:
            test_files = image_files


    if transform == 'complete_minmax[-1,1]':
        tio_bool = True
    else:
        tio_bool = False
    
    # Define transformation
    transform_dict = {
        'base': transforms.Compose([
        ]),
        'minmax[-1,1]': transforms.Compose([
            transforms.Lambda(lambda x: torch.stack([(channel - channel.min()) / (channel.max() - channel.min()) * 2 - 1 for channel in x]))
        ]),
        'complete_minmax[-1,1]': tio.Compose([
            tio.RandomFlip(axes=1, p=0.5),
            tio.RandomAffine(scales=(0.8, 1.2), degrees=(-15, 15), p=0.5),
            tio.RandomGamma(log_gamma=(0.5), p=0.5),
            tio.RescaleIntensity((0, 1)),  # optional
            tio.Lambda(lambda x: torch.stack([channel * 2 - 1 for channel in x]))  # convert [0,1] → [-1,1] per channel
        ]),
    }
    
    transform = transform_dict.get(transform, None)

    if not test:
        trainset = create_dataset(train_files, 'train', data_path, modalities, transform, tio_bool, dataset if blosc2 else None, blosc2)
        validset = create_dataset(valid_files, 'valid', data_path, modalities, transform, tio_bool, dataset if blosc2 else None, blosc2)

        if blosc2:
            trainset = SliceDataset(trainset, mode='train')
            validset = SliceDataset(validset, mode='valid')

        trainLoader = create_dataloader(trainset, batch_size, shuffle=False)
        validLoader = create_dataloader(validset, batch_size, shuffle=False)

        return trainLoader, validLoader 
    else:
        testset = create_dataset(test_files, 'test', data_path, modalities, transform, dataset if blosc2 else None, blosc2)

        if blosc2:
            test_slices, seg_slices= torch.stack(convert_volumes_to_slices(testset, 'test'))
            testset = TensorDataset(test_slices)
            seg = TensorDataset(seg_slices) 

        testLoader = create_dataloader(testset, batch_size, shuffle=False)
        segLoader = create_dataloader(seg, batch_size, shuffle=False)

        return testLoader, segLoader