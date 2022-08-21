from typing import Tuple, Optional, Dict

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset
from utils import rle2mask, resize_image
from PIL import Image
import pandas as pd
from torchvision.transforms import RandomHorizontalFlip, RandomResizedCrop


class TrainDataset(Dataset):
    def __init__(self, df_train: pd.DataFrame, image_shape: Tuple[int, int] = (256, 256),
                 dataset_dir: str = "/content/"):
        self.df_train = df_train
        self.dataset_dir = dataset_dir
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.image_shape = image_shape

    def __getitem__(self, index: int):
        h, w = self.df_train['img_height'].iloc[index], self.df_train['img_width'].iloc[index]
        path = f"{self.dataset_dir}train_images/{self.df_train['id'].iloc[index]}.tiff"
        image = Image.open(path)
        array = resize_image(np.array(image), target_shape=self.image_shape)
        image_tensor = torch.FloatTensor(array).to(self.device).permute(2, 0, 1)
        output_mask = rle2mask(mask_rle=self.df_train['rle'].iloc[index], shape=(h, w))
        output_mask_resized = resize_image(input_image=output_mask, target_shape=self.image_shape)
        target_tensor = torch.FloatTensor(output_mask_resized).to(self.device).unsqueeze(0)
        return {"input": image_tensor,
                "target": target_tensor}

    def __len__(self):
        return self.df_train.shape[0]


class TestDataset(Dataset):
    def __init__(self, df_test: pd.DataFrame, image_shape: Tuple[int, int] = (256, 256),
                 dataset_dir: str = "/content/"):
        self.df_test = df_test
        self.dataset_dir = dataset_dir
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.image_shape = image_shape

    def __getitem__(self, index: int):
        h, w = self.df_test['img_height'].iloc[index], self.df_test['img_width'].iloc[index]
        path = f"{self.dataset_dir}test_images/{self.df_test['id'].iloc[index]}.tiff"
        image = Image.open(path)
        array = resize_image(np.array(image), target_shape=self.image_shape)
        image_tensor = torch.FloatTensor(array).to(self.device).permute(2, 0, 1)
        return {"input": image_tensor}

    def __len__(self):
        return self.df_test.shape[0]


class PreTrainDataset(Dataset):
    def __init__(self, df_train: pd.DataFrame, image_shape: Tuple[int, int] = (256, 256),
                 dataset_dir: str = "/content/", transforms: Optional = None):
        self.df_train = df_train
        self.dataset_dir = dataset_dir
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.image_shape = image_shape
        if transforms is None:
            self.transform = nn.Sequential(
                RandomHorizontalFlip(),
                RandomResizedCrop((self.image_shape[0] // 2, self.image_shape[1] // 2))
            ).to(self.device)
        else:
            self.transform = transforms

    def _transform_tensors(self, result: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        input_image = result['input']
        input_target = result['target']
        concat_result = torch.cat((input_image, input_target), dim=0)
        transformed_result = self.transform(concat_result)
        return {"input": transformed_result[:3],
                "target": transformed_result[3].unsqueeze(0)}

    def __getitem__(self, index: int):
        h, w = self.df_train['img_height'].iloc[index], self.df_train['img_width'].iloc[index]
        path = f"{self.dataset_dir}train_images/{self.df_train['id'].iloc[index]}.tiff"
        image = Image.open(path)
        array = resize_image(np.array(image), target_shape=self.image_shape)
        image_tensor = torch.FloatTensor(array).to(self.device).permute(2, 0, 1)
        output_mask = rle2mask(mask_rle=self.df_train['rle'].iloc[index], shape=(h, w))
        output_mask_resized = resize_image(input_image=output_mask, target_shape=self.image_shape)
        target_tensor = torch.FloatTensor(output_mask_resized).to(self.device).unsqueeze(0)
        return self._transform_tensors({"input": image_tensor, "target": target_tensor})

    def __len__(self):
        return self.df_train.shape[0]
