"""Datasets."""

from __future__ import annotations

import glob
import os
from typing import Literal

from datasets import Dataset, Image
from torch.utils.data import DataLoader

from ptfid.data import normalize, resize


def create_dataset(
    image_folder: str,
    batch_size: int,
    image_size: tuple[int, int] = (299, 299),
    resize_name: Literal['clean', 'torch', 'tensorflow', 'pillow'] = 'tensorflow',
    normalize_name: Literal['torch', 'clip', 'inception', 'custom'] = 'inception',
    mean: list[float] | float | None = None,
    std: list[float] | float | None = None,
    num_workers: int = 8,
    pin_memory: bool = True,
) -> DataLoader:
    """Create dataset for extracting features.

    The default values assume that the feature extractor is 'inceptionv3'.

    Args:
    ----
        image_folder (str): Dir to images.
        batch_size (int): Batch size.
        image_size (tuple[int, int], optional): Size to resize images to. Default: (299, 299).
        resize_name (Literal['clean', 'torch', 'tensorflow', 'pillow'], optional): Resize method name.
            - 'tensorflow': tensorflow v1 compatible resize. Prefered when using 'inceptionv3' as the feature extractor.
            - 'clean': Resize presented in Clean-FID. Recommended for most cases.
            - 'torch': Resize image in `torch.Tensor` using `torchvision.transforms.v2.functional.resize()`.
            - 'pillow': Resize image using `Image.Image.resize()`.
            Default: 'tensorflow'.
        normalize_name (Literal['torch', 'clip', 'inception', 'custom'], optional): Normalization method name.
            - 'inception': (x - 0.5) / 0.5. Scale [0, 1] to [-1, 1].
            - 'torch': Normalize data using ImageNet mean std.
            - 'clip': Normalize data using mean std used to train CLIP models.
            - 'custom': Use custom mean std.
            Default: 'inception'.
        mean (list[float] | float | None, optional): Mean. Default: None.
        std (list[float] | float | None, optional): Std. Default: None.
        num_workers (int, optional): Number of workers for DataLoader. Default: 8.
        pin_memory (bool, optional): Pin memory for DataLoader. Default: True.

    Returns:
    -------
        DataLoader: Data loader.

    """
    image_paths = glob.glob(os.path.join(image_folder, '*'))
    image_paths = list(filter(os.path.isfile, image_paths))

    dataset = Dataset.from_dict({'image': image_paths})
    dataset = dataset.sort('image')

    dataset = dataset.cast_column('image', Image())

    resizer = resize.get_resize(resize_name)
    normalizer = normalize.get_normalize(normalize_name, v2=True, mean=mean, std=std)

    def transform_sample(sample):
        images = sample['image']
        images = [resizer(image, size=image_size) for image in images]
        images = [normalizer(image) for image in images]
        sample['image'] = images
        return sample

    dataset = dataset.with_transform(transform_sample)
    dataset = DataLoader(
        dataset, batch_size=batch_size, shuffle=False, drop_last=False, pin_memory=pin_memory, num_workers=num_workers
    )

    return dataset
