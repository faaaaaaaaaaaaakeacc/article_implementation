import matplotlib.pyplot as plt
import numpy as np
from skimage import color
import torch


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
