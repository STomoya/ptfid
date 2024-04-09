"""CLI."""

# ruff: noqa: D103
from __future__ import annotations

import pprint
from enum import Enum
from typing import Optional

import typer
from typing_extensions import Annotated

from ptfid.logger import get_logger
from ptfid.ptfid import calculate_metrics_from_folders


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


def main(
    dataset_dir1: Annotated[str, typer.Argument(help='Dir to dataset.')],
    dataset_dir2: Annotated[str, typer.Argument(help='Dir to dataset.')],
    feature_extractor: Annotated[str, typer.Option(help='Feature extractor name.')] = 'inceptionv3',
    # Metric flags.
    fid: Annotated[bool, typer.Option(help='Flag for FID.')] = True,
    kid: Annotated[bool, typer.Option(help='Flag for KID.')] = False,
    pr: Annotated[bool, typer.Option(help='Flag for P&R.')] = False,
    dc: Annotated[bool, typer.Option(help='Flag for D&C.')] = False,
    pppr: Annotated[bool, typer.Option(help='Flag for PP&PR.')] = False,
    toppr: Annotated[bool, typer.Option(help='Flag for TopP&R.')] = False,
    # Metric parameters.
    eps: Annotated[float, typer.Option(help='epsilon to avoid zero devision.')] = 1e-12,
    fid_compute_method: Annotated[FIDComputeMethods, typer.Option(help='Method to compute FID.')] = 'efficient',
    kid_times: Annotated[float, typer.Option(help='Multiply KID by.')] = 100.0,
    kid_subsets: Annotated[int, typer.Option(help='Number of subsets to compute KID.')] = 100,
    kid_subset_size: Annotated[int, typer.Option(help='Number of samples per subset.')] = 1000,
    kid_degree: Annotated[float, typer.Option(help='degree of polynomial kernel.')] = 3.0,
    kid_gamma: Annotated[Optional[float], typer.Option(help='gamma of polynomial kernel.')] = None,
    kid_coef0: Annotated[float, typer.Option(help='coef0 of polynomial kernel.')] = 1.0,
    pr_nearest_k: Annotated[int, typer.Option(help='k for nearest neighbors.')] = 5,
    pppr_alpha: Annotated[float, typer.Option(help='Alpha for PP&PR.')] = 1.2,
    toppr_alpha: Annotated[float, typer.Option(help='Alpha for TopP&R.')] = 0.1,
    toppr_kernel: Annotated[str, typer.Option(help='Kernel for TopP&R.')] = 'cosine',
    toppr_randproj: Annotated[bool, typer.Option(help='Random projection for TopP&R.')] = True,
    toppr_f1: Annotated[bool, typer.Option(help='Compute F1-score for TopP&R.')] = True,
    seed: Annotated[int, typer.Option(help='Random state seed.')] = 0,
    # Dataset parameters.
    resizer: Annotated[Resizers, typer.Option(help='Resize method.')] = 'tensorflow',
    normalizer: Annotated[Normalizers, typer.Option(help='Normalize method.')] = 'inception',
    batch_size: Annotated[int, typer.Option(help='Batch size.')] = 32,
    mean: Annotated[tuple[float, float, float], typer.Option(help='Mean for custom normalizer.')] = (None, None, None),
    std: Annotated[tuple[float, float, float], typer.Option(help='Std for custom normalizer.')] = (None, None, None),
    num_workers: Annotated[int, typer.Option(help='Number of workers.')] = 8,
    # Feature extraction arguments.
    device: Annotated[Devices, typer.Option(help='Device.')] = 'cuda',
    dataset1_is_real: Annotated[bool, typer.Option(help='Switch real dataset.')] = True,
    # other arguments.
    log_file: Annotated[Optional[str], typer.Option(help='File to output logs.')] = None,
    result_file: Annotated[str, typer.Option(help='JSON file to save results to.')] = 'results.json',
):
    """Calculate generative metrics given two image folders."""
    logger = get_logger(filename=log_file)

    local_vars = locals()
    local_vars.pop('logger')
    logger.debug('Arguments:\n' + pprint.pformat(local_vars, sort_dicts=False))

    results = calculate_metrics_from_folders(
        dataset_dir1=dataset_dir1,
        dataset_dir2=dataset_dir2,
        feature_extractor=feature_extractor,
        fid=fid,
        kid=kid,
        pr=pr,
        dc=dc,
        pppr=pppr,
        toppr=toppr,
        eps=eps,
        fid_compute_method=fid_compute_method,
        kid_times=kid_times,
        kid_subsets=kid_subsets,
        kid_subset_size=kid_subset_size,
        kid_degree=kid_degree,
        kid_gamma=kid_gamma,
        kid_coef0=kid_coef0,
        pr_nearest_k=pr_nearest_k,
        pppr_alpha=pppr_alpha,
        toppr_alpha=toppr_alpha,
        toppr_kernel=toppr_kernel,
        toppr_randproj=toppr_randproj,
        toppr_f1=toppr_f1,
        seed=seed,
        resizer=resizer,
        normalizer=normalizer,
        batch_size=batch_size,
        mean=mean,
        std=std,
        num_workers=num_workers,
        device=device,
        cache_features=False,  # We don't need to cache features when exc as cmd.
        dataset1_is_real=dataset1_is_real,
        result_file=result_file,
    )

    results_str = pprint.pformat(results)
    logger.info(f'All results:\n{results_str}')


if __name__ == '__main__':
    typer.run(main)
