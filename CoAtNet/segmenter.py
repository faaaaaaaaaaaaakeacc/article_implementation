import torch
import torch.nn as nn
import math
from math import sqrt


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


class Segmenter(nn.Module):
    """Segmenter Model."""

    def __init__(self):
        """Init Segmenter Model."""
        super().__init__()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.patch_shape = 8
        self.transformer1 = nn.Transformer(d_model=3 * self.patch_shape ** 2).encoder
        self.transformer2 = nn.Transformer(d_model=3 * self.patch_shape ** 2).encoder
        self.embedding_1 = nn.Parameter(torch.ones(3 * self.patch_shape ** 2).to(self.device))
        self.embedding_2 = nn.Parameter(torch.ones(3 * self.patch_shape ** 2).to(self.device))
        self.pe = PositionalEncoding(3 * self.patch_shape ** 2)
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.1)

    def _patch_images(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image: torch.Tensor, shape (b, 3, 256, 256)

        Returns
        -------
        torch.Tensor, shape (b, sequence_length, 3, patch_shape, patch_shape)
        """
        batch_shape = image.shape[0]
        output = []
        for i in range(image.shape[0]):
            output.append(
                image[i].unfold(0, 3, 3).unfold(1, self.patch_shape, self.patch_shape).unfold(2, self.patch_shape,
                                                                                              self.patch_shape))
        return torch.cat(output, 0).reshape(batch_shape, -1, 3, self.patch_shape, self.patch_shape)

    def _unpatch_images(self, image: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        image: torch.Tensor, shape (b, seq_length, 3 * patch_shape * patch_shape)

        Returns
        -------
        torch.Tensor, shape(b, 3, img, img)
        """
        img_size = image.shape[1] * image.shape[2] / 3
        img_size = int(sqrt(img_size) + 1e-5)
        result = torch.zeros((image.shape[0], 3, img_size, img_size)).to(self.device)
        image = image.reshape(image.shape[0], -1, 3, img_size, img_size)
        patch_size = int(sqrt(image.shape[1]) + 1e-5)
        image = image.reshape(image.shape[0], patch_size, patch_size, 3, img_size, img_size)
        for i in range(patch_size):
            for j in range(patch_size):
                l_x = i * img_size
                r_x = (i + 1) * img_size
                l_y = j * img_size
                r_y = (j + 1) * img_size
                result[:, :, l_x:r_x, l_y:r_y] = image[:, i, j, :, :, :]
        return result

    def forward(self, input_image: torch.Tensor):
        """Forward method.

        Parameters
        ----------
        input_image: torch.Tensor, shape (b, 3, img_size, img_size)
        """
        patches = self._patch_images(input_image)
        patches = patches.reshape(patches.shape[0], patches.shape[1], -1)
        patches = self.pe(patches)
        encoded_patches = self.transformer1(patches)
        encoded_patches = self.dropout(encoded_patches)
        batch_embedding_1 = torch.cat([self.embedding_1.unsqueeze(0).unsqueeze(0)] * input_image.shape[0], 0)
        encoded_patches = torch.cat([encoded_patches, batch_embedding_1], 1)
        encoded_patches = self.pe(encoded_patches)
        encoded_patches = self.transformer2(encoded_patches)
        encoded_patches = self.dropout(encoded_patches)
        original_seq = encoded_patches.shape[1] - 1
        patched_output_images = encoded_patches[:, :original_seq, :]
        segmentation_tensor = encoded_patches[:, original_seq:, :]
        segmentation_tensor = torch.cat([segmentation_tensor] * patched_output_images.shape[1], 1)

        output_transformers = patched_output_images * segmentation_tensor

        result = self._unpatch_images(output_transformers)
        result = torch.mean(result, 1).unsqueeze(1)
        return self.sigmoid(result)
