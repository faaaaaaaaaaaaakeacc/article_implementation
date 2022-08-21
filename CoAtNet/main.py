from typing import Tuple, Optional, Dict

from PIL import Image
import pandas as pd
from torchvision.transforms import RandomHorizontalFlip, RandomResizedCrop

from tqdm.auto import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from segmentation_models_pytorch.losses import DiceLoss
from typing import Dict, List
from sklearn.metrics import accuracy_score

import matplotlib.pyplot as plt
import numpy as np
from skimage import color


def mask2rle(img) -> str:
    pixels = img.T.flatten()
    pixels = np.concatenate([[0], pixels, [0]])
    runs = np.where(pixels[1:] != pixels[:-1])[0] + 1
    runs[1::2] -= runs[::2]
    return ' '.join(str(x) for x in runs)


def rle2mask(mask_rle, shape=(3000, 3000)):
    s = mask_rle.split()
    starts, lengths = [np.asarray(x, dtype=int) for x in (s[0::2], s[1::2])]
    starts -= 1
    ends = starts + lengths
    img = np.zeros(shape[0] * shape[1], dtype=np.int32)
    for lo, hi in zip(starts, ends):
        img[lo:hi] = 1
    return img.reshape(shape).T


def resize_image(input_image: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    image_original = Image.fromarray(input_image)
    reshaped_image = image_original.resize(target_shape)
    array_transformed = np.array(reshaped_image)
    return array_transformed


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


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.neck = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.upconv2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Dropout(0.1)
        )
        self.upconv1 = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input_image):
        hidden1 = self.conv1(input_image)  # 64, 128, 128
        hidden2 = self.conv2(hidden1)  # 64, 64, 64
        output = self.neck(hidden2)
        output = torch.cat((output, hidden2), dim=1)
        output = self.upconv2(output)
        output = torch.cat((output, hidden1), dim=1)
        return self.upconv1(output)


def train(model: nn.Module,
          optimizer: torch.optim,
          train_loader: DataLoader,
          criterion: nn.Module,
          device: torch.device,
          num_epochs: int = 10,
          log_window: int = 20):
    for epoch in trange(num_epochs):
        pbar = tqdm(train_loader)
        sum_loss, cnt_loss = 0, 0
        cur_sum_loss, cur_cnt_loss = 0, 0
        for batch in pbar:
            optimizer.zero_grad()
            input_tensor = batch['input'].to(device)
            target_tensor = batch['target'].to(device)
            output = model(input_tensor)
            loss = criterion(output, target_tensor)
            loss.backward()
            sum_loss += loss.item()
            cnt_loss += 1
            cur_sum_loss += loss.item()
            cur_cnt_loss += 1
            description = f"Mean total loss: {round(sum_loss / cnt_loss, 3)} | mean window loss: {round(cur_sum_loss / cur_cnt_loss, 3)}"
            pbar.set_description(description)
            if cur_cnt_loss == log_window:
                cur_sum_loss, cur_cnt_loss = 0, 0
            optimizer.step()
        torch.save(model.state_dict(), "output_model")


def load_weights(model: nn.Module, PATH: str) -> nn.Module:
    model.load_state_dict(torch.load(PATH))
    return model


@torch.no_grad()
def inference_valid(model: nn.Module,
                    valid_loader: DataLoader,
                    device: torch.device,
                    criterion: nn.Module = DiceLoss('binary'),
                    threshold: float = 0.5,
                    eps: float = 1e-5
                    ) -> Dict[str, float]:
    sum_loss, cnt_loss = 0, 0
    sum_acc, cnt_acc = 0, 0
    for batch in valid_loader:
        input_tensor = batch['input'].to(device)
        output = model(input_tensor)

        output_np = output.detach().cpu().numpy().reshape(-1)
        target_np = batch['target'].detach().cpu().numpy().reshape(-1)

        sum_acc += accuracy_score(y_true=target_np, y_pred=output_np)
        cnt_acc += 1

        loss = criterion(output, batch['target'].to(device))
        sum_loss += loss.item()
        cnt_loss += 1

    return {"loss": sum_loss / cnt_loss,
            "accuracy": sum_acc / cnt_acc}


@torch.no_grad()
def get_rle_predictions(model: nn.Module,
                        test_dataset: Dataset,
                        device: torch.device,
                        threshold: float = 0.5
                        ) -> List[str]:
    answer = []
    for batch in test_dataset:
        input_tensor = batch['input'].unsqueeze(0).to(device)
        output = model(input_tensor).cpu().detach().numpy()
        rle = mask2rle(output)
        answer.append(rle)
    return answer


def show_masked_img(img, mask, title=''):
    mask = (mask - mask.min()) / (mask.max() - mask.min())
    mask = np.nan_to_num(mask)
    mask = np.round(mask)
    mask = mask.reshape(img.shape[:2])

    fig, ax = plt.subplots(1, 3, figsize=(9, 3))
    fig.suptitle(title, fontsize=16)

    ax[0].imshow(mask)
    ax[0].set_title('Mask')
    ax[1].imshow(img)
    ax[1].set_title('Image')
    ax[2].imshow(color.label2rgb(mask, img,
                                 bg_label=0, bg_color=(1., 1., 1.), alpha=0.25))
    ax[2].set_title('Masked Image')
    plt.show()


def plot_batch(output: torch.Tensor, target: torch.Tensor):
    output_cpu = output.permute(0, 2, 3, 1).cpu().detach().numpy()
    target_cpu = target.permute(0, 2, 3, 1).cpu().detach().numpy()
    for img, msk in zip(target_cpu, output_cpu):
        show_masked_img(img, msk, "Sample")


class TransUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.01),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Dropout(0.1)

        )

        self.encoder = nn.Transformer(nhead=4).encoder

        self.up1 = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.01),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.Dropout(0.1)
        )

        self.up2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.01),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.Dropout(0.1)
        )
        self.last = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 1, 3, 1, 1)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, input: torch.Tensor):
        hidden1 = self.conv1(input)  # b, 64, 128, 128
        hidden2 = self.conv2(hidden1)  # b, 128, 64, 64
        hidden3 = self.conv3(hidden2)  # b, 256, 32, 32
        batch_size = hidden3.shape[0]
        image_shape = tuple(hidden3.shape)
        transformer_patch = hidden3.reshape(batch_size, -1, 512)
        output_transformer = self.encoder(transformer_patch)
        unpatched_batch = output_transformer.reshape(image_shape)  # b, 256, 32, 32
        output = torch.cat((unpatched_batch, hidden3), 1)  # b, 512, 32, 32
        output = self.up1(output)  # b, 128, 64, 64
        output = torch.cat((output, hidden2), 1)  # b, 256, 64, 64
        output = self.up2(output)  # b, 64, 128, 128
        output = torch.cat((output, hidden1), 1)  # b, 64, 128, 128
        output = self.last(output)
        return self.sigmoid(output)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.neck = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        self.upconv2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Dropout(0.1)
        )
        self.upconv1 = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 1, 3, 1, 1),
            nn.Sigmoid()
        )

    def forward(self, input_image):
        hidden1 = self.conv1(input_image)  # 64, 128, 128
        hidden2 = self.conv2(hidden1)  # 64, 64, 64
        output = self.neck(hidden2)
        output = torch.cat((output, hidden2), dim=1)
        output = self.upconv2(output)
        output = torch.cat((output, hidden1), dim=1)
        return self.upconv1(output)


class UnetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = smp.UnetPlusPlus(in_channels=3, encoder_depth=3, decoder_channels=[64, 32, 16]).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor):
        output = self.model(X)
        return self.sigmoid(output)


from torch.utils.data import DataLoader, Subset, Dataset
from torch.utils.data import random_split


def split_dataset(dataset: Subset, train_part: float) -> Tuple[Subset, Subset]:
    train_len = int(len(dataset) * train_part)
    lengths = [train_len, len(dataset) - train_len]
    train, test = random_split(dataset, lengths=lengths)
    return (train, test)
