import os
import numpy as np
import torch


def read_data(dataset, volume_key, modalities, modality_map={'t1c': 0, 't1n': 1, 't2f': 2, 't2w': 3}):
    mapped = [modality_map[modality] for modality in modalities]

    data, seg = dataset.load_case(volume_key)

    data = torch.tensor(np.array(data), dtype=torch.float32)
    seg = torch.tensor(np.array(seg), dtype=torch.float32)

    data = data[mapped, ...] 

    return data, seg 