"""Utils."""

from __future__ import annotations

import glob
import json
import os
import random
import shutil
from contextlib import contextmanager
from typing import Any, ClassVar

import numpy as np
import torch


@contextmanager
def local_seed_numpy(seed: int, enabled: bool = True):
    """Locally set the seed of numpy.

    Args:
    ----
        seed (int): Seed.
        enabled (bool, optional): Enable local seed if True. Default: True.

    """
    if enabled:
        random_state = np.random.get_state()
        np.random.seed(seed)

    yield

    if enabled:
        np.random.set_state(random_state)


@contextmanager
def local_seed_builtin(seed: int, enabled: bool = True):
    """Locally set the seed of builtin random.

    Args:
    ----
        seed (int): Seed.
        enabled (bool, optional): Enable local seed if True. Default: True.

    """
    if enabled:
        random_state = random.getstate()
        hash_seed = os.environ.get('PYTHONHASHSEED')

        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)

    yield

    if enabled:
        random.setstate(random_state)
        if hash_seed is None:
            os.environ.pop('PYTHONHASHSEED')
        else:
            os.environ['PYTHONHASHSEED'] = hash_seed


@contextmanager
def local_seed_torch(seed: int, enabled: bool = True):
    """Locally set the seed of torch.

    Args:
    ----
        seed (int): Seed.
        enabled (bool, optional): Enable local seed if True. Default: True.

    """
    cuda_is_available = torch.cuda.is_available()

    if enabled:
        random_state_cpu = torch.get_rng_state()
        if cuda_is_available:
            # Currently we only work on one process.
            random_state_gpu = torch.cuda.get_rng_state()

        torch.manual_seed(seed)
        if cuda_is_available:
            torch.cuda.manual_seed(seed)

    yield

    if enabled:
        torch.set_rng_state(random_state_cpu)
        if cuda_is_available:
            torch.cuda.set_rng_state(random_state_gpu)


@contextmanager
def local_deterministic(enabled: bool = True):
    """Localy set to use deterministic algorithms.

    Args:
    ----
        enabled (bool, optional): Default: True.

    """
    cuda_is_available = torch.cuda.is_available()
    if enabled:
        deterministic = torch.are_deterministic_algorithms_enabled()
        if cuda_is_available:
            cudnn_benchmark = torch.backends.cudnn.benchmark

        torch.use_deterministic_algorithms(mode=True, warn_only=True)
        if cuda_is_available:
            torch.backends.cudnn.benchmark = False

    yield

    if enabled:
        torch.use_deterministic_algorithms(deterministic)
        if cuda_is_available:
            torch.backends.cudnn.benchmark = cudnn_benchmark


@contextmanager
def local_seed(
    seed: int,
    enabled: bool = True,
    deterministic: bool = True,
    seed_builtin: int | None = None,
    seed_numpy: int | None = None,
    seed_torch: int | None = None,
):
    """Locally set the seed.

    Args:
    ----
        seed (int): Seed.
        enabled (bool, optional): Enable local seed if True. Default: True.
        deterministic (bool, optional): Enable deterministic algorithm. Default: True.
        seed_builtin (int, optional): Seed for `random`. If not given `seed` is used.
        seed_numpy (int, optional): Seed for `numpy`. If not given `seed` is used.
        seed_torch (int, optional): Seed for `torch`. If not given `seed` is used.

    """
    seed_builtin = seed_builtin or seed
    seed_numpy = seed_numpy or seed
    seed_torch = seed_torch or seed

    with (
        local_seed_builtin(seed_builtin, enabled),
        local_seed_numpy(seed_numpy, enabled),
        local_seed_torch(seed_torch, enabled),
        local_deterministic(deterministic),
    ):
        yield


def save_json(obj: dict[str, Any], filename: str, avoid_overwrite: bool = True) -> None:
    """Save obj to JSON file.

    Args:
    ----
        obj (dict[str, Any]): object to save.
        filename (str): Pathlike string.
        avoid_overwrite (bool, optional): Avoid overwriting file by renaming the file.

    """
    if '/' in filename:
        folder = '/'.join(filename.split('/')[:-1])
        # skip './<filename>.json' pattern.
        if folder != '.':
            os.makedirs(folder, exist_ok=True)

    if avoid_overwrite and os.path.exists(filename):
        # rename old file to avoid overwriting the file.
        paths = glob.glob(filename + '.*')
        index = len(paths)
        shutil.move(filename, filename + f'.{index:02d}')

    with open(filename, 'w') as fp:
        json.dump(obj, fp, indent=2)


class FeatureCache:
    """Cache."""


class InMemoryFeatureCache(FeatureCache):
    """In memory cache."""

    _cache: ClassVar = {}

    @classmethod
    def set(cls, folder, model, features):
        """Set features."""
        cls._cache[(folder, model)] = features

    @classmethod
    def get(cls, folder, model):
        """Get features."""
        return cls._cache.get((folder, model))
