"""CLI."""

# ruff: noqa: D103
from __future__ import annotations

from enum import Enum

import typer
from typing_extensions import Annotated


class Resizers(str, Enum):
    """Resizer enum."""

    clean = 'clean'
    torch = 'torch'
    tensorflow = 'tensorflow'
    pillow = 'pillow'


class Normalizers(str, Enum):
    """Normalizer enum."""

    torch = 'torch'
    clip = 'clip'
    inception = 'inception'
    custom = 'custom'


class Devices(str, Enum):
    """Device enum."""

    cpu = 'cpu'
    cuda = 'cuda'


def main(
    dataset_dir1: Annotated[str, typer.Argument(help='Dir to dataset.')],
    dataset_dir2: Annotated[str, typer.Argument(help='Dir to dataset.')],
    feature_extractor: Annotated[str, typer.Option(help='Feature extractor name.')] = 'inceptionv3',
    resizer: Annotated[Resizers, typer.Option(help='Resize method.')] = 'tensorflow',
    normalizer: Annotated[Normalizers, typer.Option(help='Normalize method.')] = 'inception',
    batch_size: Annotated[int, typer.Option(help='Batch size.')] = 32,
    mean: Annotated[tuple[float, float, float], typer.Option(help='Mean for custom normalizer.')] = (None, None, None),
    std: Annotated[tuple[float, float, float], typer.Option(help='Std for custom normalizer.')] = (None, None, None),
    num_workers: Annotated[int, typer.Option(help='Number of workers.')] = 8,
    device: Annotated[Devices, typer.Option(help='Device.')] = 'cuda',
    dataset1_is_real: Annotated[bool, typer.Option(help='Switch real dataset.')] = True,
):
    pass
    # TODO


if __name__ == '__main__':
    typer.run(main)
