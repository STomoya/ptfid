"""ResNet50."""

from __future__ import annotations

from collections import OrderedDict

import torch.nn as nn
from torchvision.models import resnet50

from ptfid.feature.utils import load_state_dict_from_url


def get_resnet50_swav_model() -> tuple[nn.Module, tuple[int, int, int]]:
    """Get ResNet model with weights trained using SwAV."""
    state_dict = load_state_dict_from_url('https://dl.fbaipublicfiles.com/deepcluster/swav_800ep_pretrain.pth.tar')
    model = resnet50()
    model.fc = nn.Identity()

    new_state_dict = OrderedDict()
    for name, value in state_dict.items():
        name = name.replace('module.', '')  # noqa: PLW2901
        if not name.startswith(('projection_head.', 'prototypes.')):
            new_state_dict[name] = value

    model.load_state_dict(new_state_dict)
    return model, (3, 224, 224)
