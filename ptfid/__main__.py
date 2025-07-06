"""CLI."""

# ruff: noqa: D103
from __future__ import annotations

import logging
import pprint
from typing import Optional

import typer
from typing_extensions import Annotated

from ptfid.config import (
    Devices,
    FeatureExtractionParameters,
    FIDComputeMethods,
    FIDParameters,
    Interpolations,
    KIDParameters,
    LogLevels,
    MetricFlags,
    MetricParameters,
    Normalizers,
    PPPRParameters,
    PRParameters,
    Resizers,
    TopPRParameters,
)
from ptfid.logger import get_logger
from ptfid.ptfid import calculate_metrics_from_folders

app = typer.Typer(
    rich_markup_mode='markdown',
    pretty_exceptions_enable=False,
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=True,
)


@app.command()
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
    interpolation: Annotated[Interpolations, typer.Option(help='Interpolation mode.')] = 'bicubic',
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
    log_level: Annotated[LogLevels, typer.Option(help='Logging level.')] = 'warning',
):
    """Calculate generative metrics given two image folders."""
    logger = get_logger(filename=log_file, logging_level=getattr(logging, log_level.upper()))

    local_vars = locals()
    local_vars.pop('logger')
    logger.debug('Arguments:\n' + pprint.pformat(local_vars, sort_dicts=False))

    metric_flags = MetricFlags(fid=fid, kid=kid, pr=pr, dc=dc, pppr=pppr, toppr=toppr)
    metric_params = MetricParameters(
        fid=FIDParameters(eps=eps, method=fid_compute_method),
        kid=KIDParameters(
            times=kid_times,
            subsets=kid_subsets,
            subset_size=kid_subset_size,
            degree=kid_degree,
            gamma=kid_gamma,
            coef0=kid_coef0,
        ),
        pr=PRParameters(nearest_k=pr_nearest_k),
        pppr=PPPRParameters(alpha=pppr_alpha),
        toppr=TopPRParameters(
            alpha=toppr_alpha,
            kernel=toppr_kernel,
            randproj=toppr_randproj,
            f1=toppr_f1,
        ),
        seed=seed,
        feat1_is_real=True,
    )
    feature_params = FeatureExtractionParameters(
        feature_extractor=feature_extractor,
        device=device,
        cache_features=False,
        image_size=None,
        dataset1_is_real=dataset1_is_real,
        resizer=resizer,
        interpolation=interpolation,
        normalizer=normalizer,
        batch_size=batch_size,
        mean=mean,
        std=std,
        num_workers=num_workers,
    )

    results = calculate_metrics_from_folders(
        dataset_dir1=dataset_dir1,
        dataset_dir2=dataset_dir2,
        metrics=metric_flags,
        metric_params=metric_params,
        feature_params=feature_params,
        result_file=result_file,
    )

    results_str = pprint.pformat(results)
    logger.info(f'All results:\n{results_str}')


def cli():
    typer.run(main)


if __name__ == '__main__':
    app()
