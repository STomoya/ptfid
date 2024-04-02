"""P&R, D&C, PP&PR.

https://github.com/clovaai/generative-evaluation-prdc/blob/a0daab888cc7acb86699dddc04d3667bed8f785c/prdc/prdc.py
https://github.com/kdst-team/Probablistic_precision_recall/blob/a6c70d22552eb9379ec6eea29d4f9cfe2e1d765d/metric/pp_pr.py
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import pairwise_distances


def compute_pairwise_distance(data_x: np.ndarray, data_y: np.ndarray = None) -> np.ndarray:
    """pdist.

    Args:
    ----
        data_x (np.ndarray): data.
        data_y (np.ndarray, optional): data. Default: None.

    Returns:
    -------
        np.ndarray: pairwise distances.

    """
    if data_y is None:
        data_y = data_x
    dists = pairwise_distances(data_x, data_y, metric='euclidean', n_jobs=8)
    return dists


def compute_all_pairwise_distances(
    real_features: np.ndarray, fake_features: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute all patterns of pairwise distances used to calculate P&R and PP&PR."""
    distances_real_fake = pairwise_distances(real_features, fake_features)
    return (
        compute_pairwise_distance(real_features),
        compute_pairwise_distance(fake_features),
        distances_real_fake,
        np.transpose(distances_real_fake),
    )


def get_kth_value(unsorted: np.ndarray, k: int, axis: int = -1) -> np.ndarray:
    """Get kth value.

    Args:
    ----
        unsorted (np.ndarray): Unsorted data.
        k (int): Number of values.
        axis (int, optional): Default: -1

    Returns:
    -------
        np.ndarray: kth values along the designated axis.

    """
    indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
    k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
    kth_values = k_smallests.max(axis=axis)
    return kth_values


def precision(real_nn_distances: np.ndarray, distances_real_fake: np.ndarray) -> np.ndarray:
    """Precision."""
    return (distances_real_fake < np.expand_dims(real_nn_distances, axis=1)).any(axis=0).mean()


def recall(fake_nn_distances: np.ndarray, distances_real_fake: np.ndarray) -> np.ndarray:
    """Recall."""
    return (distances_real_fake < np.expand_dims(fake_nn_distances, axis=0)).any(axis=1).mean()


def density(real_nn_distances: np.ndarray, distances_real_fake: np.ndarray, nearest_k: int) -> np.ndarray:
    """Density."""
    return (1.0 / float(nearest_k)) * (distances_real_fake < np.expand_dims(real_nn_distances, axis=1)).sum(
        axis=0
    ).mean()


def coverage(real_nn_distance: np.ndarray, distance_real_fake: np.ndarray) -> dict[str, np.ndarray]:
    """Coverage."""
    return (distance_real_fake.min(axis=1) < real_nn_distance).mean()


def get_scoring_rule_psr(distances: np.ndarray, nearest_avg: np.ndarray, alpha: float = 1.2) -> np.ndarray:
    """Compute scoring rule of PSR."""
    alpha_nearest = alpha * nearest_avg
    out_of_nearest = distances >= alpha_nearest
    psr = 1 - distances / alpha_nearest
    psr[out_of_nearest] = 0.0
    psr = np.prod(1.0 - psr, axis=0)
    return psr


def get_PSR_XY(
    real_nn_distances: np.ndarray,
    fake_nn_distances: np.ndarray,
    distances: np.ndarray,
    distances_t: np.ndarray,
    alpha: float = 1.2,
):
    """Compute variables for computing PP&PR."""
    k_nearest_real = real_nn_distances.mean()
    k_nearest_fake = fake_nn_distances.mean()
    PSR_real = get_scoring_rule_psr(distances, k_nearest_real, alpha)
    PSR_fake = get_scoring_rule_psr(distances_t, k_nearest_fake, alpha)
    return PSR_real, PSR_fake


def p_precision(PSR_real):
    """P-Precision."""
    return np.mean(1.0 - PSR_real)


def p_recall(PSR_fake):
    """P-Recall."""
    return np.mean(1.0 - PSR_fake)


def calculate_pr(
    features1: np.ndarray,
    features2: np.ndarray,
    nearest_k: int,
    pppr_alpha: float = 1.2,
    feat1_is_real: bool = True,
    pr: bool = True,
    dc: bool = True,
    pppr: bool = True,
) -> dict[str, float]:
    """Calculate P&R and/or PP&PR.

    Args:
    ----
        features1 (np.ndarray): Features of dataset 1.
        features2 (np.ndarray): Features of dataset 2.
        nearest_k (int): k for nearest neighbor.
        pppr_alpha (float, optional): Alpha for PP&PR. Defaults to 1.2.
        feat1_is_real (bool, optional): Switch whether `features1` or `features2` is extracted from real samples.
            Default: True.
        pr (bool, optional): Flag. False to skip calculation. Defaults to True.
        dc (bool, optional): Flag. False to skip calculation. Defaults to True.
        pppr (bool, optional): Flag. False to skip calculation. Defaults to True.

    Returns:
    -------
        dict[str, float]: score.

    """
    if not pr and not dc:
        return {}

    real_features, fake_features = (features1, features2) if feat1_is_real else (features2, features1)

    distances_real = compute_pairwise_distance(real_features)
    distances_fake = compute_pairwise_distance(fake_features)
    distances_real_fake = compute_pairwise_distance(real_features, fake_features)
    distances_fake_real = np.transpose(distances_real_fake)

    real_nn_distances = get_kth_value(distances_real, k=nearest_k + 1, axis=-1)
    fake_nn_distances = get_kth_value(distances_fake, k=nearest_k + 1, axis=-1)

    results = {}
    if pr:
        p = precision(real_nn_distances, distances_real_fake)
        r = recall(fake_nn_distances, distances_real_fake)
        results.update({'precision': p, 'recall': r})
    if dc:
        d = density(real_nn_distances, distances_real_fake, nearest_k)
        c = coverage(real_nn_distances, distances_real_fake)
        results.update({'density': d, 'coverage': c})
    if pppr:
        PSR_real, PSR_fake = get_PSR_XY(
            real_nn_distances, fake_nn_distances, distances_real_fake, distances_fake_real, alpha=pppr_alpha
        )
        pp = p_precision(PSR_real)
        pr_ = p_recall(PSR_fake)
        results.update({'p_precision': pp, 'p_recall': pr_})

    return results
