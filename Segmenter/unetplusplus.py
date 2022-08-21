import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class UnetPlusPlus(nn.Module):
    def __init__(self):
        super().__init__()
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.model = smp.UnetPlusPlus(in_channels=3, encoder_depth=3, decoder_channels=[64, 32, 16]).to(device)
        self.sigmoid = nn.Sigmoid()

    def forward(self, X: torch.Tensor):
        output = self.model(X)
        return self.sigmoid(output)
