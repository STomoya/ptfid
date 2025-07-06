"""Resize."""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize, to_tensor
from torchvision.transforms.v2.functional import resize as resize_v2
from torchvision.transforms.v2.functional import to_dtype, to_image

from ptfid.config import Interpolations, Resizers
from ptfid.const import PIL_INTERP_MODE, TORCH_INTERP_MODE
from ptfid.data.interpolate_compat_tensorflow import interpolate_bilinear_2d_like_tensorflow1x
from ptfid.data.interpolate_compat_tensorflow_orig import (
    interpolate_bilinear_2d_like_tensorflow1x as interpolate_bilinear_2d_like_tensorflow1x_orig,
)


def set_interpolation(mode: Interpolations = 'bicubic'):
    """Set interpolation mode for libraries. The tf_resize can only be set to bilinear."""
    global PIL_INTERP_MODE, TORCH_INTERP_MODE  # noqa: PLW0603
    assert mode in ['bicubic', 'bilinear']

    if mode == 'bicubic':
        PIL_INTERP_MODE = Image.BICUBIC
        TORCH_INTERP_MODE = InterpolationMode.BICUBIC
    elif mode == 'bilinear':
        PIL_INTERP_MODE = Image.BILINEAR
        TORCH_INTERP_MODE = InterpolationMode.BILINEAR


def clean_resize(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """Clean resize.

    from: https://github.com/layer6ai-labs/dgm-eval/blob/d57a6d2d6bd613332dd21696994afff8efa78575/dgm_eval/resizer.py
    """

    def resize_single_channel(x):
        img = Image.fromarray(x, mode='F')
        img = img.resize(size[::-1], resample=PIL_INTERP_MODE)
        return np.asarray(img).clip(0, 255).reshape(*size, 1)

    x = np.array(image.convert('RGB')).astype(np.float32)
    x = [resize_single_channel(x[:, :, idx]) for idx in range(3)]
    x = np.concatenate(x, axis=2).astype(np.float32)

    x = to_image(x)
    x = x / 255.0
    return x


def clean_resize2(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """Clean resize without converting data multiple times. The output is equivalent with `clean_resize`."""
    image_splits = image.split()
    new_image = []
    for image_split in image_splits:
        channel = image_split.convert('F')
        channel = channel.resize(size[::-1], resample=PIL_INTERP_MODE)
        new_image.append(np.asarray(channel).clip(0, 255).reshape(*size, 1))
    image = np.concatenate(new_image, axis=2)

    image = to_image(image)
    image = image / 255.0
    return image


def pil_resize(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """Resize image using PIL resize."""
    image = image.resize(size[::-1], resample=PIL_INTERP_MODE)
    image = to_image(image)
    image = to_dtype(image, dtype=torch.float32, scale=True)
    return image


def torch_resize_legacy(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """Resize image using torch interpolation without antialias."""
    image = to_tensor(image)
    image = resize(image, size, interpolation=TORCH_INTERP_MODE, antialias=False)
    return image


def torch_resize(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """Resize image using torch interpolation with antialias."""
    image = to_image(image)
    image = to_dtype(image, torch.float32, scale=True)
    image = resize_v2(image, size, interpolation=TORCH_INTERP_MODE, antialias=True)
    return image


def tf_resize_orig(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """Tensorflow compatibale bilinear interpolation."""
    image = to_image(image)
    image = to_dtype(image, dtype=torch.float32, scale=True)
    image = image.unsqueeze(0)
    image = interpolate_bilinear_2d_like_tensorflow1x_orig(image, size, align_corners=False)
    image = image.squeeze()
    return image


def tf_resize(image: Image.Image, size: tuple[int, int]) -> torch.Tensor:
    """Tensorflow compatibale bilinear interpolation."""
    image = to_image(image)
    image = to_dtype(image, dtype=torch.float32, scale=True)
    image = interpolate_bilinear_2d_like_tensorflow1x(image, size, align_corners=False)
    return image


def get_resize(
    name: Resizers,
    interpolation: Interpolations = 'bicubic',
) -> Callable[[Image.Image, tuple[int, int]], torch.Tensor]:
    """Create resizing function based on name.

    Args:
    ----
        name (Resizers): Name of the resizer.
        interpolation (Interpolations, optional): Interpolation mode. Default: 'bicubic'.

    Returns:
    -------
        Callable: Function for resize.

    """
    set_interpolation(interpolation)
    return {
        Resizers.clean: clean_resize2,
        Resizers.torch: torch_resize,
        Resizers.tensorflow: tf_resize,
        Resizers.pillow: pil_resize,
    }[name]
