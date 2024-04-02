"""Utils."""

from __future__ import annotations

from contextlib import contextmanager
from typing import ClassVar

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
