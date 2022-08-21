import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

def get_tensor_image_from_path(path):
    img = Image.open(path).resize((256, 256))
    convert_tensor = transforms.ToTensor()
    return convert_tensor(img)

class ImageDetectionDataset(Dataset):
    def __init__(self, df, DIR, delta = 2):
        self.df = df
        self.DIR = DIR
        self.augmented_list = [] # (path, xcenter, ycenter, w, h, confidence)
        self.masks = []
        for data_pos in range(self.df.shape[0]):
            for position in range(4):
                for i in range(-delta, delta):
                    image_path = self.DIR + df['image'].iloc[data_pos]
                    frame = [df['xmin'].iloc[data_pos], df['xmax'].iloc[data_pos], df['ymin'].iloc[data_pos], df['ymax'].iloc[data_pos]]
                    frame[position] += i
                    confidence = IoU(df['xmin'].iloc[data_pos], df['xmax'].iloc[data_pos], df['ymin'].iloc[data_pos], df['ymax'].iloc[data_pos],
                                     frame[0], frame[1], frame[2], frame[3])
                    x_center = (frame[0] + frame[1]) / 2
                    y_center = (frame[2] + frame[3]) / 2
                    w = frame[1] - frame[0]
                    h = frame[3] - frame[2]

                    x_center *= (256 / 676)
                    w *= (256 / 676)
                    y_center *= (256 / 380)
                    h *= (256 / 380)

                    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

                    self.augmented_list.append((image_path, x_center, y_center, w, h, confidence))
                    expected_mask = torch.zeros((8, 8)).to(device)

                    for i in range(8):
                        for j in range(8):
                            expected_mask[i][j] =  IoU(i * 32, (i * 32) + 31, j * 32, (j * 32) + 31,
                                    x_center - w / 2,  x_center + w / 2,
                                    y_center - h / 2, y_center + w / 2)

                    self.masks.append(expected_mask)

    def __len__(self):
        return len(self.augmented_list)

    def __getitem__(self, position):
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        image_tensor = get_tensor_image_from_path(self.augmented_list[position][0]).to(device)
        target_tensor = torch.FloatTensor([self.augmented_list[position][1],
                                           self.augmented_list[position][2],
                                           self.augmented_list[position][3],
                                           self.augmented_list[position][4],
                                           self.augmented_list[position][5]]).to(device)

        return {'X': image_tensor, 'y': target_tensor, 'mask': self.masks[position]}


class DETR(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(in_channels=32, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3, stride=1, padding=1),
        )
        # 8 * 512
        self.transformer = nn.Transformer()
        self.positional_encoding = nn.Parameter(torch.randn(8, 512))
        self.bboxes = nn.Parameter(torch.randn(20, 512))
        self.fc = nn.Sequential(nn.Linear(512, 32),
                                nn.ReLU(),
                                nn.Linear(32, 5))

    def forward(self, input):
        hidden = self.cnn(input)
        hidden = hidden.reshape((-1, 8, 512))
        hidden += self.positional_encoding
        attent = self.transformer.encoder(hidden)
        bboxes_after_transformer = self.transformer.decoder(self.bboxes, attent)
        final_boxes = self.fc(bboxes_after_transformer)
        return final_boxes


def SQuare(xmin, xmax, ymin, ymax):
    return (ymax - ymin) * (xmax - xmin)

def IoU(xmin, xmax, ymin, ymax, Xmin, Xmax, Ymin, Ymax):
    x_in_min = max(xmin, Xmin)
    x_in_max = min(xmax, Xmax)
    y_in_min = max(ymin, Ymin)
    y_in_max = min(ymax, Ymax)

    intersection_square = SQuare(x_in_min, x_in_max, y_in_min, y_in_max)
    union_square = SQuare(xmin, xmax, ymin, ymax) +  SQuare(Xmin, Xmax, Ymin, Ymax) - intersection_square

    return intersection_square / union_square

def IoUTensors(a: torch.tensor, b: torch.tensor) -> torch.tensor:
    xmin = a[0] - a[2] / 2
    xmax = a[0] + a[2] / 2
    ymin = a[1] - a[3] / 2
    ymax = a[1] + a[3] / 2

    Xmin = b[0] - b[2] / 2
    Xmax = b[0] + b[2] / 2
    Ymin = b[1] - b[3] / 2
    Ymax = b[1] + b[3] / 2
    return IoU(xmin, xmax, ymin, ymax, Xmin, Xmax, Ymin, Ymax)

def single_dim_hungarian_loss(input: torch.tensor, target: torch.tensor, l1: float, l2: float) -> torch.tensor:
    return min([l1 * (1 - input[i][4]) + l2 * IoU(input[i], target) for i in range(input.shape[0])])

