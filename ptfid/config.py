"""Pydantic config models."""

from enum import Enum

import torch
import torch.nn as nn
from pydantic import BaseModel, ConfigDict, field_validator


class FIDComputeMethods(str, Enum):
    """FID compute methods."""

    original = 'original'
    efficient = 'efficient'
    gpu = 'gpu'


class Resizers(str, Enum):
    """Resizer enum."""

    clean = 'clean'
    torch = 'torch'
    tensorflow = 'tensorflow'
    pillow = 'pillow'


class Interpolations(str, Enum):
    """Interpolation methods."""

    bilinear = 'bilinear'
    bicubic = 'bicubic'


class Normalizers(str, Enum):
    """Normalizer enum."""

    imagenet = 'imagenet'
    openai = 'openai'
    inception = 'inception'
    custom = 'custom'


class Devices(str, Enum):
    """Device enum."""

    cpu = 'cpu'
    cuda = 'cuda'


class LogLevels(str, Enum):
    """Log level enum."""

    debug = 'debug'
    info = 'info'
    warning = 'warning'
    error = 'error'
    critical = 'critical'


class MetricFlags(BaseModel):
    """Metric flags."""

    fid: bool = True
    kid: bool = False
    pr: bool = False
    dc: bool = False
    pppr: bool = False
    toppr: bool = False


class FIDParameters(BaseModel):
    """FID parameters."""

    eps: float = 1e-12
    method: FIDComputeMethods = 'efficient'


class KIDParameters(BaseModel):
    """KID parameters."""

    times: float = 100.0
    subsets: int = 100
    subset_size: int = 1000
    degree: float = 3.0
    gamma: float | None = None
    coef0: float = 1.0


class PRParameters(BaseModel):
    """P&R parameters."""

    nearest_k: int = 5


class PPPRParameters(BaseModel):
    """PP&PR parameters."""

    alpha: float = 1.2


class TopPRParameters(BaseModel):
    """TopP&R parameters."""

    alpha: float = 0.1
    kernel: str = 'cosine'
    randproj: bool = True
    f1: bool = True


class MetricParameters(BaseModel):
    """Metric parameters."""

    fid: FIDParameters = FIDParameters()
    kid: KIDParameters = KIDParameters()
    pr: PRParameters = PRParameters()
    pppr: PPPRParameters = PPPRParameters()
    toppr: TopPRParameters = TopPRParameters()
    seed: int = 0
    feat1_is_real: bool = True


class FeatureExtractionParameters(BaseModel):
    """Feature extraction parameters."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Feature extractor related
    feature_extractor: str | nn.Module = 'inceptionv3'
    device: Devices = 'cuda'
    cache_features: bool = False
    image_size: int | None = None
    dataset1_is_real: bool = True
    # Dataset related
    resizer: Resizers = 'tensorflow'
    interpolation: Interpolations = 'bicubic'
    batch_size: int = 32
    mean: tuple[float | None, float | None, float | None] = (None, None, None)
    std: tuple[float | None, float | None, float | None] = (None, None, None)
    normalizer: Normalizers = 'inception'
    num_workers: int = 8

    @field_validator('device', mode='before')
    @classmethod
    def _check_device(cls, v: Devices) -> Devices:
        if v == Devices.cuda and not torch.cuda.is_available():
            raise ValueError('CUDA is not available. Please use CPU.')
        return v

    @field_validator('normalizer', mode='after')
    @classmethod
    def _check_normalizer(cls, v: Normalizers, info: field_validator) -> Normalizers:
        if v == Normalizers.custom and (None in info.data['mean'] or None in info.data['std']):
            raise ValueError('`mean` and `std` must be set when `normalizer` is `custom`.')
        return v
