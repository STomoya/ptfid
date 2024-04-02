"""Efficient computation of Frechet distance.

https://github.com/layer6ai-labs/dgm-eval/blob/d57a6d2d6bd613332dd21696994afff8efa78575/dgm_eval/metrics/fd.py#L79-L89
"""

import numpy as np
from scipy import linalg


def calculate_frechet_distance(mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray) -> np.ndarray:
    """Efficient computation of Frechet distance.

    Original docstring:
    ------------------
        A more efficient computation of FD as proposed at the following link:
        https://www.reddit.com/r/MachineLearning/comments/12hv2u6/d_a_better_way_to_compute_the_fr%C3%A9chet_inception/

    Args:
    ----
        mu1 (np.ndarray): mean of features1.
        sigma1 (np.ndarray): covariance of features1.
        mu2 (np.ndarray): mean of features1.
        sigma2 (np.ndarray): covariance of features1.

    Returns:
    -------
        np.ndarray: score.

    """
    a = np.square(mu1 - mu2).sum()
    b = sigma1.trace() + sigma2.trace()
    c = np.real(linalg.eigvals(sigma1 @ sigma2) ** 0.5).sum()
    return a + b - 2 * c
