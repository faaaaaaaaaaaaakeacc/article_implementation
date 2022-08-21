import os
import json
import numpy as np
from torch.utils.data import Dataset
import torch
import nibabel as nib
from typing import Dict, Tuple, Any


class TLDatasetCource(Dataset):
    """Init dataset class for dataset from competition."""

    def __init__(self, path: str):
        """Init dataset.

        Parameters
        ----------
        path: str
            Path of data
        """
        super().__init__()
        labels, images = self._prepare_data(path)

        self.labels = []
        self.images = []
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for i in range(len(images)):
            images_tensor = torch.FloatTensor(images[i]).permute(2, 0, 1).to(device)
            labels_tensor = torch.FloatTensor(labels[i]).permute(2, 0, 1).to(device)
            self.images.append([])
            self.labels.append([])
            for j in range(images_tensor.shape[0]):
                self.images[-1].append(images_tensor[j].reshape(1, 512, 512).to(device))
                self.labels[-1].append(labels_tensor[j].reshape(1, 512, 512).to(device))

    @staticmethod
    def _prepare_data(path: str = '../input/tgcovid/data/data') -> Tuple[Any, Any]:
        path_images = os.path.join(path, 'images')
        path_labels = os.path.join(path, 'labels')
        with open('../input/tgcovid/training_data.json', 'r') as f:
            dict_training = json.load(f)

        images = []
        labels = []
        for entry in dict_training:
            s1 = os.path.join(path_images, entry['image'])
            sp1 = s1[:len(s1) - 3]

            s2 = os.path.join(path_labels, entry['label'])
            sp2 = s2[:len(s2) - 3]
            image = nib.load(sp1)
            label = nib.load(sp2)
            images.append(image.get_fdata().astype(np.float32))
            labels.append(label.get_fdata().astype(np.float32))
        return labels, images

    def __len__(self) -> int:
        """Computes length of dataset."""
        return len(self.images)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        """Get item with input index."""
        return {"input": self.images[index],
                "target": self.labels[index]}
