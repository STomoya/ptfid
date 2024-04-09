"""DINOv2."""

from __future__ import annotations

import warnings
from typing import Literal

import torch
import torch.nn as nn


def _get_dinov2_model(
    arch: Literal['vits14', 'vitb14', 'vitl14', 'vitg14'] = 'vitl14',
) -> tuple[nn.Module, tuple[int, int, int]]:
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        model = torch.hub.load('facebookresearch/dinov2', f'dinov2_{arch}')
    return model, (224, 224)


def get_dinov2_s_model() -> tuple[nn.Module, tuple[int, int]]:
    """ViT-S-14."""
    return _get_dinov2_model('vits14')


def get_dinov2_b_model() -> tuple[nn.Module, tuple[int, int]]:
    """ViT-B-14."""
    return _get_dinov2_model('vitb14')


def get_dinov2_l_model() -> tuple[nn.Module, tuple[int, int]]:
    """ViT-L-14."""
    return _get_dinov2_model('vitl14')


def get_dinov2_g_model() -> tuple[nn.Module, tuple[int, int]]:
    """ViT-G-14."""
    return _get_dinov2_model('vitg14')


def get_dinov2_model() -> tuple[nn.Module, tuple[int, int]]:
    """Create default."""
    return get_dinov2_l_model()
