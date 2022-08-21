import numpy as np
from typing import Tuple
from PIL import Image


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
