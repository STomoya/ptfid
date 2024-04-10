"""Normlization."""

from __future__ import annotations

from typing import Callable, Sequence

import torch
from torchvision.transforms.functional import normalize as normalize_fn
from torchvision.transforms.v2.functional import normalize as normalizev2_fn

TORCH_IMAGENET_MEAN = [0.485, 0.456, 0.406]
TORCH_IMAGENET_STD = [0.229, 0.224, 0.225]

CLIP_IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_IMAGENET_STD = [0.26862954, 0.26130258, 0.27577711]

# NOTE: Original implementation uses (x - 128) / 128, where x is uint8.
#       Here, we scale the image in float32 [0,1] for simplicity, which will NOT result in bit exact results,
#       but should be fine enough.
TF_INCEPTION_MEAN = [128.0 / 255.0]
TF_INCEPTION_STD = [128.0 / 255.0]


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
            return _normalize_fn(tensor, TORCH_IMAGENET_MEAN, TORCH_IMAGENET_STD)

    elif lib == 'openai':

        def normalize(tensor):
            return _normalize_fn(tensor, CLIP_IMAGENET_MEAN, CLIP_IMAGENET_STD)

    elif lib == 'inception':

        def normalize(tensor):
            return _normalize_fn(tensor, TF_INCEPTION_MEAN, TF_INCEPTION_STD)

    elif lib == 'custom':
        assert mean is not None and std is not None
        if v2:
            assert isinstance(mean, Sequence) and isinstance(std, Sequence)

        def normalize(tensor):
            return _normalize_fn(tensor, mean, std)

    else:
        raise Exception(f'Unknown normlization type "{lib}"')

    return normalize
