"""consts."""

from __future__ import annotations

from PIL import Image
from torchvision.transforms import InterpolationMode

# arguments to reproduce other implementations
FID_ARGUMENTS = dict(
    fid_compute_method='original',
)
CLEANFID_ARGUMENTS = dict(
    resizer='clean',
)
FDD_ARGUMENTS = dict(
    feature_extractor='dinov2',
    resizer='pillow',
    normalizer='imagenet',
)
SWAVFID_ARGUMENTS = dict(
    feature_extractor='resnet50',
    resizer='pillow',
    interpolation='bilinear',
    normalizer='imagenet',
)
COMMON_ARGUMENTS = dict(
    fid=FID_ARGUMENTS,
    cleanfid=CLEANFID_ARGUMENTS,
    fdd=FDD_ARGUMENTS,
    swavfid=SWAVFID_ARGUMENTS,
)


def get_reproduction_args(name: str) -> dict[str, str]:
    """Get arguments for reproducing implementations.

    Args:
    ----
        name (str): Name of the implementation.

    Raises:
    ------
        Exception: Unknown name.

    Returns:
    -------
        dict[str, str]: arguments.

    Example:
    -------
        >>> args = get_reproduction_args('swavfid')
        >>> results = calculate_metrics_from_folders(dir1, dir2, fid=True, **args)

    """
    args = COMMON_ARGUMENTS.get(name)
    if args is None:
        raise Exception(f'Unknown argument name "{name}"')
    return args


# normalization
TORCH_IMAGENET_MEAN = [0.485, 0.456, 0.406]
TORCH_IMAGENET_STD = [0.229, 0.224, 0.225]

CLIP_IMAGENET_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_IMAGENET_STD = [0.26862954, 0.26130258, 0.27577711]

# NOTE: Original implementation uses (x - 128) / 128, where x is uint8.
#       Here, we scale the image in float32 [0,1] for simplicity, which will NOT result in bit exact results,
#       but should be fine enough.
TF_INCEPTION_MEAN = [128.0 / 255.0]
TF_INCEPTION_STD = [128.0 / 255.0]


# resizer interpolation.
PIL_INTERP_MODE = Image.BICUBIC
TORCH_INTERP_MODE = InterpolationMode.BICUBIC


# logger
LOGGER_NAME = 'ptfid'
