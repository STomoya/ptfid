"""MMD.

from:
- https://github.com/layer6ai-labs/dgm-eval/blob/d57a6d2d6bd613332dd21696994afff8efa78575/dgm_eval/metrics/mmd.py
- https://github.com/toshas/torch-fidelity/blob/ca69a1016419dd7cd876faba8d803438c77fec8c/torch_fidelity/metric_kid.py
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics.pairwise import polynomial_kernel

from ptfid.utils import local_seed_numpy


def mmd2(K_XX, K_XY, K_YY):
    """https://github.com/dougalsutherland/opt-mmd/blob/master/two_sample/mmd.py.

    changed to not compute the full kernel matrix at once
    """
    m = K_XX.shape[0]
    assert K_XX.shape == (m, m)
    assert K_XY.shape == (m, m)
    assert K_YY.shape == (m, m)

    diag_X = np.diagonal(K_XX)
    diag_Y = np.diagonal(K_YY)

    Kt_XX_sums = K_XX.sum(axis=1) - diag_X
    Kt_YY_sums = K_YY.sum(axis=1) - diag_Y
    K_XY_sums_0 = K_XY.sum(axis=0)

    Kt_XX_sum = Kt_XX_sums.sum()
    Kt_YY_sum = Kt_YY_sums.sum()
    K_XY_sum = K_XY_sums_0.sum()

    mmd2 = (Kt_XX_sum + Kt_YY_sum) / (m * (m - 1))
    mmd2 -= 2 * K_XY_sum / (m * m)
    return mmd2


def compute_polynomial_mmd(feat_r, feat_gen, degree=3, gamma=None, coef0=1):
    """Use  k(x, y) = (gamma <x, y> + coef0)^degree."""
    X = feat_r
    Y = feat_gen

    K_XX = polynomial_kernel(X, degree=degree, gamma=gamma, coef0=coef0)
    K_YY = polynomial_kernel(Y, degree=degree, gamma=gamma, coef0=coef0)
    K_XY = polynomial_kernel(X, Y, degree=degree, gamma=gamma, coef0=coef0)

    return mmd2(K_XX, K_XY, K_YY)


def calculate_kid(
    features1: np.ndarray,
    features2: np.ndarray,
    num_subsets: int = 100,
    subset_size: int = 1000,
    degree: float = 3.0,
    gamma: float | None = None,
    coef0: float = 1.0,
    times: float = 100.0,
    seed: int = 0,
) -> dict[str, float]:
    """Calc KID.

    Args:
    ----
        features1 (np.ndarray): features from dataset 1.
        features2 (np.ndarray): features from dataset 2.
        num_subsets (int, optional): Number of subsets to compute KID. Default: 100.
        subset_size (int, optional): Subset size to compute KID. Default: 1000.
        degree (float, optional): degree for polynomial kernel. Default: 3.0.
        gamma (float | None, optional): gamma for polynomial kernel. Default: None.
        coef0 (float, optional): coef0 for polynomial kernel. Default: 1.0.
        times (float, optional): Value to multiply KID after calculation. x100 is commonly used. Default: 100.0.
        seed (int, optional): Defaults to 0.

    Returns:
    -------
        dict[str, float]: score.

    """
    subset_size = min((features1.shape[0], features2.shape[0], subset_size))
    mmds = np.zeros(num_subsets)

    with local_seed_numpy(seed):
        for i in num_subsets:
            feat1_ = features1[np.random.choice(len(features1), subset_size, replace=False)]
            feat2_ = features2[np.random.choice(len(features2), subset_size, replace=False)]
            o = compute_polynomial_mmd(feat1_, feat2_, degree=degree, gamma=gamma, coef0=coef0)
            mmds[i] = o

    if times == 1:
        return {'kid.mean': mmds.mean(), 'kid.std': mmds.std()}
    else:
        return {f'kid.mean.x{int(times):d}': mmds.mean() * times, f'kid.std.x{int(times):d}': mmds.std()}
