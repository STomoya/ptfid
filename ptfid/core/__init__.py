"""Core."""

from __future__ import annotations

import numpy as np

from ptfid.config import MetricFlags, MetricParameters
from ptfid.const import METRIC_FLAGS_DEFAULT, METRIC_PARAMS_DEFAULT
from ptfid.core.fid import calculate_frechet_distance
from ptfid.core.kid import calculate_kid
from ptfid.core.pr import calculate_pr
from ptfid.core.toppr import calculate_toppr
from ptfid.logger import Timer, get_logger


def calculate_metrics_from_features(
    features1: np.ndarray,
    features2: np.ndarray,
    metrics: MetricFlags = METRIC_FLAGS_DEFAULT,
    metric_params: MetricParameters = METRIC_PARAMS_DEFAULT,
) -> dict[str, float]:
    """Calculate generative model metric scores based on extracted features.

    Args:
    ----
        features1 (np.ndarray): Features of dataset 1. This dataset can be either real or fake. Use `feat1_is_real` to
            indicate which is real.
        features2 (np.ndarray): Features of dataset 2. This dataset can be either real or fake. Use `feat1_is_real` to
            indicate which is real.
        metrics (MetricFlags, optional): Metric flags. Defaults to `MetricFlags()`.\
            - `fid`: Flag. False to skip calculation. Defaults to `True`. \
            - `kid`: Flag. False to skip calculation.  Defaults to `False`. \
            - `pr`: Flag. False to skip calculation.  Defaults to `True`. \
            - `dc`: Flag. False to skip calculation.  Defaults to `False`. \
            - `pppr`: Flag. False to skip calculation.  Defaults to `False`. \
            - `toppr`: Flag. False to skip calculation.  Defaults to `False`.
        metric_params (MetricParameters, optional): Metric parameters. Defaults to `MetricParameters()`. \
            **FID** \
                - `eps`: eps for avoiding division by zero. Defaults to `1e-12`. \
                - `method`: Determine how to compute Frechet distance. Must be one of 'original', \
                    'efficient', 'gpu'. Defaults to `'efficient'`. \
            **KID** \
                - `times`: Value to multiply KID after calculation. x100 is commonly used. Defaults to `100.0`. \
                - `subsets`: Number of subsets to compute KID. Defaults to `100`. \
                - `subset_size`: Subset size to compute KID. Defaults to `1000`. \
                - `degree`: degree for polynomial kernel. Defaults to `3.0`. \
                - `gamma`: gamma for polynomial kernel. Defaults to `None`. \
                - `coef0`: coef0 for polynomial kernel. Defaults to `1.0`. \
            **P&R** \
                - `nearest_k`: k for nearest neighbor. Defaults to `5`. \
            **PP&PR** \
                - `alpha`: Alpha for PP&PR. Defaults to `1.2`. \
            **TopP&R** \
                - `alpha`: Alpha for TopP&TopR. Defaults to `0.1`. \
                - `kernel`: kernel for TopP&TopR. Defaults to `'cosine'`. \
                - `randproj`: Perform random projection. Defaults to `True`. \
                - `f1`: Compute F1-score. Defaults to `True`. \
            **Other** (top level of `MetricParameters`) \
                - `feat1_is_real`: Switch whether `features1` or `features2` is extracted from real samples. \
                    Defaults to `True`. \
                - `seed`: Random seed. Defaults to `0`.

    Returns:
    -------
        dict[str, float]: calculated scores.

    """
    logger = get_logger()
    timer = Timer()

    results = {}

    if metrics.fid:
        logger.info('FID...')
        timer.start()

        _fid = calculate_frechet_distance(
            features1=features1,
            features2=features2,
            eps=metric_params.fid.eps,
            method=metric_params.fid.method,
        )

        duration = timer.done()
        logger.info(f'  result: {_fid}')
        logger.debug(f'  duration: {duration}')
        results.update(_fid)

    if metrics.kid:
        logger.info('KID...')
        timer.start()

        _kid = calculate_kid(
            features1=features1,
            features2=features2,
            num_subsets=metric_params.kid.subsets,
            subset_size=metric_params.kid.subset_size,
            degree=metric_params.kid.degree,
            gamma=metric_params.kid.gamma,
            coef0=metric_params.kid.coef0,
            times=metric_params.kid.times,
            seed=metric_params.seed,
        )

        duration = timer.done()
        logger.info(f'  result: {_kid}')
        logger.debug(f'  duration: {duration}')
        results.update(_kid)

    if any((metrics.pr, metrics.dc, metrics.pppr)):
        logger.info('P&R, D&C, PP&PR...')
        timer.start()

        _pr = calculate_pr(
            features1=features1,
            features2=features2,
            nearest_k=metric_params.pr.nearest_k,
            pppr_alpha=metric_params.pppr.alpha,
            feat1_is_real=metric_params.feat1_is_real,
            pr=metrics.pr,
            dc=metrics.dc,
            pppr=metrics.pppr,
        )

        duration = timer.done()
        logger.info(f'  results: {_pr}')
        logger.debug(f'  duration: {duration}')
        results.update(_pr)

    if metrics.toppr:
        logger.info('TopP&R...')
        timer.start()

        _toppr = calculate_toppr(
            features1=features1,
            features2=features2,
            alpha=metric_params.toppr.alpha,
            kernel=metric_params.toppr.kernel,
            random_proj=metric_params.toppr.randproj,
            f1_score=metric_params.toppr.f1,
            feat1_is_real=metric_params.feat1_is_real,
            seed=metric_params.seed,
        )

        duration = timer.done()
        logger.info(f'  results: {_toppr}')
        logger.debug(f'  duration: {duration}')
        results.update(_toppr)

    return results
