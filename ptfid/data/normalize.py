"""Normlization."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torchvision.transforms.functional import normalize as normalize_fn
from torchvision.transforms.v2.functional import normalize as normalizev2_fn

from ptfid import const


def get_normalize(
    lib: str, v2: bool = True, mean: list[float] | None = None, std: list[float] | None = None
) -> Callable[[torch.Tensor], torch.Tensor]:
    """Create and return a function for normalization."""
    _normalize_fn = normalizev2_fn if v2 else normalize_fn

    if isinstance(mean, float):
        mean = [mean]

    if isinstance(std, float):
        std = [std]

    if lib == 'imagenet':

        def normalize(tensor):
            return _normalize_fn(tensor, const.TORCH_IMAGENET_MEAN, const.TORCH_IMAGENET_STD)

    elif lib == 'openai':

        def normalize(tensor):
            return _normalize_fn(tensor, const.CLIP_IMAGENET_MEAN, const.CLIP_IMAGENET_STD)

    elif lib == 'inception':

        def normalize(tensor):
            return _normalize_fn(tensor, const.TF_INCEPTION_MEAN, const.TF_INCEPTION_STD)

    elif lib == 'custom':
        assert mean is not None and std is not None
        if v2:
            assert isinstance(mean, Sequence) and isinstance(std, Sequence)

        def normalize(tensor):
            return _normalize_fn(tensor, mean, std)

    else:
        raise Exception(f'Unknown normlization type "{lib}"')

    return normalize
