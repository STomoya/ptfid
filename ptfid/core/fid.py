"""FID."""

from __future__ import annotations

import numpy as np

from ptfid.core import efficient_fid, fid_gpu, original


def feature_statistics(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance of the given feature array.

    Args:
    ----
        features (np.ndarray): the extracted features.

    Returns:
    -------
        np.ndarray: mean
        np.ndarray: covariance

    """
    mean = np.mean(features, axis=0)
    sigma = np.cov(features, rowvar=False)
    return mean, sigma


def calculate_frechet_distance(
    features1: np.ndarray, features2: np.ndarray, eps=1e-8, method: str = 'efficient'
) -> dict[str, float]:
    """Calc FID from given feature vectors.

    Args:
    ----
        features1 (np.ndarray): features from dataset 1.
        features2 (np.ndarray): features from dataset 2.
        eps (float, optional): Defaults to 1e-8.
        method (str, optional): Defaults to 'efficient'.

    Returns:
    -------
        dict: score.

    """
    mu1, sigma1 = feature_statistics(features1)
    mu2, sigma2 = feature_statistics(features2)

    if method == 'original':
        score = original.calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps)
    elif method == 'efficient':
        score = efficient_fid.calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
    elif method == 'gpu':
        score = fid_gpu.calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps)

    return {'fid': score}
