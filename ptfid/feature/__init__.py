"""Feature."""

from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn

from ptfid.feature.clip import (
    get_clip_arch_model,
    get_clip_b16_model,
    get_clip_b32_model,
    get_clip_bigG14_model,
    get_clip_l14_model,
    get_clip_model,
)
from ptfid.feature.dinov2 import (
    get_dinov2_b_model,
    get_dinov2_g_model,
    get_dinov2_l_model,
    get_dinov2_model,
    get_dinov2_s_model,
)
from ptfid.feature.inception_ts import get_inceptionv3_model
from ptfid.feature.resnet50 import get_resnet50_swav_model
from ptfid.feature.timm_models import get_timm_model
from ptfid.logger import get_logger

PTFID_MODEL_VARIANT: dict[str, dict[str, Callable[[], nn.Module]]] = {
    'dinov2': {
        'default': get_dinov2_model,
        'vits14': get_dinov2_s_model,
        'vitb14': get_dinov2_b_model,
        'vitl14': get_dinov2_l_model,
        'vitg14': get_dinov2_g_model,
    },
    'clip': {
        'default': get_clip_model,
        'vitb32': get_clip_b32_model,
        'vitb16': get_clip_b16_model,
        'vitl14': get_clip_l14_model,
        'vitbigG14': get_clip_bigG14_model,
    },
    'inceptionv3': {
        'default': get_inceptionv3_model,
    },
    'resnet50': {
        'default': get_resnet50_swav_model,
    },
}


def _split_name(name: str) -> tuple[str, str, str]:
    name_splits_ = name.split(':')

    if len(name_splits_) == 1:
        source = 'ptfid'
    elif len(name_splits_) == 2:
        source = name_splits_[0]
        name = name_splits_[1]
    else:
        raise Exception(f'Model name format should be (source:)model(.variant). got {name}')

    if source not in ['ptfid', 'clip', 'timm']:
        raise Exception(f'Source must be one of ["ptfid", "timm", "clip"]. Got {source}')

    name_splits_ = name.split('.')

    if len(name_splits_) == 1:
        model = name_splits_[0]
        variant = 'default'
    elif len(name_splits_) == 2:
        model = name_splits_[0]
        variant = name_splits_[1]
    else:
        raise Exception(f'Model name format should be (source:)model(.variant). got {name}')

    return source, model, variant


def get_feature_extractor(name: str, device: torch.device) -> tuple[nn.Module, tuple[int, int]]:
    """Create feature extractor model.

    The format of the name is:
        `(<source>:)<model>(.<variant>)` () is optional, <> must be replaced by user.
    <model> is the name of the model. For example `'inceptionv3'` uses Inceptionv3, `'resnet50'` uses ResNet50 trained
    using SwAV. <variant> is the weight variant. If not given the default model is used. If <source> is given, the model
    will downloaded from the source. <source> can be one of `'timm'` or `'clip'`. For `'timm'` the further name is used
    to load the model. For example `'timm:convnext_tiny.fb_in1k'` will call
    `timm.create_model('convnext_tiny.fb_in1k', pretrained=True)`.

    Args:
    ----
        name (str): name of the feature extractor.
        device (torch.device): Device.

    Returns:
    -------
        nn.Module: Feature extractor.

    """
    logger = get_logger()
    logger.info(f'Input feature extractor name: "{name}"')

    source, model, variant = _split_name(name)

    if source == 'ptfid':
        extractor, input_size = PTFID_MODEL_VARIANT[model][variant]()

    if source == 'clip':
        extractor, input_size = get_clip_arch_model(model, variant)

    if source == 'timm':
        extractor, input_size = get_timm_model('.'.join([model, variant]) if variant != 'default' else model)

    extractor.to(device)
    extractor.eval()
    extractor.requires_grad_(False)

    logger.info(f'Feature extractor class: "{extractor.__class__.__qualname__}"')
    logger.debug(f'Input image size: {input_size}')
    logger.debug(f'Architecture:\n{extractor}')

    return extractor, input_size
