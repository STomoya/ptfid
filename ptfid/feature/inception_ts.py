"""Inceptionv3."""

# ruff: noqa: D107, D102, D103

from __future__ import annotations

import contextlib
import os

import torch
import torch.nn as nn
from stutil.download import download_url


@contextlib.contextmanager
def disable_gpu_fuser_on_pt19():
    """On PyTorch 1.9 a CUDA fuser bug prevents the Inception JIT model to run.

    See:
        https://github.com/GaParmar/clean-fid/issues/5
        https://github.com/pytorch/pytorch/issues/64062
    """
    if torch.__version__.startswith('1.9.'):
        old_val = torch._C._jit_can_fuse_on_gpu()
        torch._C._jit_override_can_fuse_on_gpu(False)
    yield
    if torch.__version__.startswith('1.9.'):
        torch._C._jit_override_can_fuse_on_gpu(old_val)


class InceptionV3W(nn.Module):
    """Inception v3 Torch Script.

    Wrapper around Inception V3 torchscript model provided here
    https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt
    """

    def __init__(self, download=True):
        super(InceptionV3W, self).__init__()
        # download the network if it is not present at the given directory
        # use the current directory by default
        filename = 'inception-2015-12-05-ts.pt'
        if download:
            download_url(
                'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/inception-2015-12-05.pt',
                root='./.cache/pyfid/weights',
                filename=filename,
            )
        path = os.path.join('./.cache/pyfid/weights', filename)
        self.base = torch.jit.load(path).eval()
        self.layers = self.base.layers

    def forward(self, x):
        bs, _, height, width = x.size()
        assert (height == 299) and (width == 299)

        with disable_gpu_fuser_on_pt19():
            features = self.layers.forward(x).view((bs, 2048))
            return features


def get_inceptionv3_model() -> tuple[nn.Module, tuple[int, int]]:
    return InceptionV3W(), (299, 299)
