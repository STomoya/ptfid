"""Calc FID on GPU.

https://github.com/jaywu109/faster-pytorch-fid/blob/e2073cbc99c2d4840d60b5654bd0ed0decadb670/fid_score_gpu.py#L154-L190
"""

from functools import partial

import numpy as np

from ptfid.core._sqrtm_gpu import np_to_gpu_tensor, sqrtm, torch_matmul_to_array


def calculate_frechet_distance(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray, eps: float = 1e-6, device: str = 'cuda'
) -> np.ndarray:
    """Calculate FID on GPU.

    Args:
    ----
        mu1 (np.ndarray): mean of features1.
        sigma1 (np.ndarray): covariance of features1.
        mu2 (np.ndarray): mean of features1.
        sigma2 (np.ndarray): covariance of features1.
        eps (float, optional): Defaults to 1e-6.
        device (str, optional): Defaults to 'cuda'.

    Raises:
    ------
        ValueError:

    Returns:
    -------
        np.ndarray: score.

    """
    array_to_tensor = partial(np_to_gpu_tensor, device)
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, 'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # Product might be almost singular
    covmean, _ = sqrtm(
        torch_matmul_to_array(array_to_tensor(sigma1), array_to_tensor(sigma2)), array_to_tensor, disp=False
    )

    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; ' 'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = sqrtm(
            torch_matmul_to_array(array_to_tensor(sigma1 + offset), array_to_tensor(sigma2 + offset)), array_to_tensor
        )

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    diff_ = array_to_tensor(diff)
    return torch_matmul_to_array(diff_, diff_) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean
