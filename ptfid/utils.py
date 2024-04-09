"""Utils."""

from __future__ import annotations

import glob
import json
import os
import shutil
from contextlib import contextmanager
from typing import Any, ClassVar

import numpy as np


@contextmanager
def local_seed(seed: int, enabled: bool = True):
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
        cls._cache[tuple(folder, model)] = features

    @classmethod
    def get(cls, folder, model):
        """Get features."""
        return cls._cache.get(tuple(folder, model))
