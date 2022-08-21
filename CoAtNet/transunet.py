import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


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
        self.pe = PositionalEncoding(512)

    def forward(self, input: torch.Tensor):
        hidden1 = self.conv1(input) # b, 64, 128, 128
        hidden2 = self.conv2(hidden1) # b, 128, 64, 64
        hidden3 = self.conv3(hidden2) # b, 256, 32, 32
        batch_size = hidden3.shape[0]
        image_shape = tuple(hidden3.shape)
        transformer_patch = hidden3.reshape(batch_size, -1, 512)
        transformer_patch = self.pe(transformer_patch)
        output_transformer = self.encoder(transformer_patch)
        unpatched_batch = output_transformer.reshape(image_shape) # b, 256, 32, 32
        output = torch.cat((unpatched_batch, hidden3), 1) # b, 512, 32, 32
        output = self.up1(output) # b, 128, 64, 64
        output = torch.cat((output, hidden2), 1) # b, 256, 64, 64
        output = self.up2(output) # b, 64, 128, 128
        output = torch.cat((output, hidden1), 1) # b, 64, 128, 128
        output = self.last(output)
        return self.sigmoid(output)






