import torch
import torch.nn as nn


class COATWiseNet(nn.Module):
    """Transformer-based neural network with convolutional positional encoding."""

    def __init__(self, img_size: int = 256):
        """Init model.

        Parameters
        ----------
        img_size: int
            shape of input image

        """
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4, 2, 1),
        )

        self.block2 = nn.Sequential(
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(32, 64, 3, 1, 1)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, 3, 1, 1)
        )

        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(64, 64, 3, 1, 1),
        )
        self.d_model = (img_size // 16) ** 2
        self.transformer = nn.Transformer(nhead=5, d_model=self.d_model).encoder

        self.decoder = nn.Sequential(
            nn.Conv2d(224, 128, 3, 1, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(32, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2, mode='nearest'),
        )

        self.last = nn.Sequential(
            nn.Conv2d(35, 16, 3, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Conv2d(16, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(),
            nn.Conv2d(8, 4, 3, 1, 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            nn.Conv2d(4, 1, 3, 1, 1)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, input_image: torch.Tensor) -> torch.Tensor:
        """Segmentes input images.

        Parameters
        ----------
        input_image: torch.Tensor
            images, shape (b, 3, img_size, img_size)

        Returns
        -------
        torch.Tensor, shape (b, 1, img_size, img_size)
        """
        features = self.block1(input_image)
        features2 = self.block2(features)
        features3 = self.block3(features2)
        features4 = self.block4(features3)

        encoded_input = torch.cat([features, features2, features3, features4], dim=1)
        original_shape = tuple(encoded_input.shape)
        encoded_input = encoded_input.reshape(encoded_input.shape[0], encoded_input.shape[1], -1)
        output = self.transformer(encoded_input)
        output = output.reshape(original_shape)
        decoded_output = self.decoder(output)

        decoded_output = torch.cat([decoded_output, input_image], dim=1)
        decoded_output = self.last(decoded_output)
        return self.sigmoid(decoded_output)
