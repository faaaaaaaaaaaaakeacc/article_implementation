import pandas as pd
from tqdm import tqdm, trange
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, Dataset
from segmentation_models_pytorch.losses import DiceLoss
from typing import Dict, List, Tuple, Sequence
from sklearn.metrics import accuracy_score
import numpy as np
from utils import mask2rle
from torch.utils.data import random_split
from utils import resize_image

def split_dataset(dataset: Subset, train_part: float) -> Tuple[Subset, Subset]:
    train_len = int(len(dataset) * train_part)
    lengths = [train_len, len(dataset) - train_len]
    train, test = random_split(dataset, lengths=lengths)
    return (train, test)


def train(model: nn.Module,
          optimizer: torch.optim,
          train_loader: DataLoader,
          criterion: nn.Module,
          device: torch.device,
          num_epochs: int = 10,
          log_window: int = 20):
    model.train()
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
    model.eval()
    sum_loss, cnt_loss = 0, 0
    sum_acc, cnt_acc = 0, 0
    for batch in valid_loader:
        input_tensor = batch['input'].to(device)
        output = model(input_tensor)

        output_np = output.detach().cpu().numpy().reshape(-1) > threshold
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
                        heights: Sequence[int],
                        widths: Sequence[int],
                        threshold: float = 0.5
                        ) -> List[str]:
    model.train()
    answer = []
    for i, batch in enumerate(test_dataset):
        input_tensor = batch['input'].unsqueeze(0).to(device)
        output = model(input_tensor).cpu().detach().numpy() > threshold
        output = resize_image(output, (heights[i], widths[i]))
        rle = mask2rle(output)
        answer.append(rle)
    return answer
