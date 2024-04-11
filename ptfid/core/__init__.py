"""Core."""

from __future__ import annotations

import numpy as np

from ptfid.core.fid import calculate_frechet_distance
from ptfid.core.kid import calculate_kid
from ptfid.core.pr import calculate_pr
from ptfid.core.toppr import calculate_toppr
from ptfid.logger import Timer, get_logger


def calculate_metrics_from_features(
    features1: np.ndarray,
    features2: np.ndarray,
    fid: bool = True,
    kid: bool = False,
    pr: bool = False,
    dc: bool = False,
    pppr: bool = False,
    toppr: bool = False,
    eps: float = 1e-12,
    fid_compute_method: str = 'efficient',
    kid_times: float = 100.0,
    kid_subsets: int = 100,
    kid_subset_size: int = 1000,
    kid_degree: float = 3.0,
    kid_gamma: float | None = None,
    kid_coef0: float = 1.0,
    pr_nearest_k: int = 5,
    pppr_alpha: float = 1.2,
    toppr_alpha: float = 0.1,
    toppr_kernel: str = 'cosine',
    toppr_randproj: bool = True,
    toppr_f1: bool = True,
    feat1_is_real: bool = True,
    seed: int = 0,
) -> dict[str, float]:
    """Calculate generative model metric scores based on extracted features.

    Args:
    ----
        features1 (np.ndarray): Features of dataset 1. This dataset can be either real or fake. Use `feat1_is_real` to
            indicate which is real.
        features2 (np.ndarray): Features of dataset 2. This dataset can be either real or fake. Use `feat1_is_real` to
            indicate which is real.
        fid (bool, optional): Flag. False to skip calculation. Default: True.
        kid (bool, optional): Flag. False to skip calculation.  Default: False.
        pr (bool, optional): Flag. False to skip calculation.  Default: True.
        dc (bool, optional): Flag. False to skip calculation.  Default: False.
        pppr (bool, optional): Flag. False to skip calculation.  Default: False.
        toppr (bool, optional): Flag. False to skip calculation.  Default: False.
        eps (float, optional): eps for avoiding division by zero. Default: 1e-12.
        fid_compute_method (str, optional): Determine how to compute Frechet distance. Must be one of 'original',
            'efficient', 'gpu'. Default: 'efficient'.
        kid_times (float, optional): Value to multiply KID after calculation. x100 is commonly used. Default: 100.0.
        kid_subsets (int, optional): Number of subsets to compute KID. Default: 100.
        kid_subset_size (int, optional): Subset size to compute KID. Default: 1000.
        kid_degree (float, optional): degree for polynomial kernel. Default: 3.0.
        kid_gamma (float | None, optional): gamma for polynomial kernel. Default: None.
        kid_coef0 (float, optional): coef0 for polynomial kernel. Default: 1.0.
        pr_nearest_k (int, optional): k for nearest neighbor. Default: 5.
        pppr_alpha (float, optional): Alpha for PP&PR. Default: 1.2.
        toppr_alpha (float, optional): Alpha for TopP&TopR. Default: 0.1.
        toppr_kernel (str, optional): kernel for TopP&TopR. Default: 'cosine'.
        toppr_randproj (bool, optional): Perform random projection. Default: True.
        toppr_f1 (bool, optional): Compute F1-score. Default: True.
        feat1_is_real (bool, optional): Switch whether `features1` or `features2` is extracted from real samples.
            Default: True.
        seed (int, optional): Random seed. Default: 0.

    Returns:
    -------
        dict[str, float]: calculated scores.

    """
    logger = get_logger()
    timer = Timer()

    results = {}

    if fid:
        logger.info('FID...')
        timer.start()

        _fid = calculate_frechet_distance(
            features1=features1,
            features2=features2,
            eps=eps,
            method=fid_compute_method,
        )

        duration = timer.done()
        logger.info(f'  result: {_fid}')
        logger.debug(f'  duration: {duration}')
        results.update(_fid)

    if kid:
        logger.info('KID...')
        timer.start()

        _kid = calculate_kid(
            features1=features1,
            features2=features2,
            num_subsets=kid_subsets,
            subset_size=kid_subset_size,
            degree=kid_degree,
            gamma=kid_gamma,
            coef0=kid_coef0,
            times=kid_times,
            seed=seed,
        )

        duration = timer.done()
        logger.info(f'  result: {_kid}')
        logger.debug(f'  duration: {duration}')
        results.update(_kid)

    if any((pr, dc, pppr)):
        logger.info('P&R, D&C, PP&PR...')
        timer.start()

        _pr = calculate_pr(
            features1=features1,
            features2=features2,
            nearest_k=pr_nearest_k,
            pppr_alpha=pppr_alpha,
            feat1_is_real=feat1_is_real,
            pr=pr,
            dc=dc,
            pppr=pppr,
        )

        duration = timer.done()
        logger.info(f'  results: {_pr}')
        logger.debug(f'  duration: {duration}')
        results.update(_pr)

    if toppr:
        logger.info('TopP&R...')
        timer.start()

        _toppr = calculate_toppr(
            features1=features1,
            features2=features2,
            alpha=toppr_alpha,
            kernel=toppr_kernel,
            random_proj=toppr_randproj,
            f1_score=toppr_f1,
            feat1_is_real=feat1_is_real,
            seed=seed,
        )

        duration = timer.done()
        logger.info(f'  results: {_toppr}')
        logger.debug(f'  duration: {duration}')
        results.update(_toppr)

    return results
