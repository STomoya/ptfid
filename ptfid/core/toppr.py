"""TopP&TopR.

https://github.com/LAIT-CVLab/TopPR/blob/bf7655136d1fc7aab231a0516f4bf7945656ff07/top_pr/top_pr.py
"""

from __future__ import annotations

import gudhi
import numpy as np
import torch
from scipy.spatial import distance
from sklearn.metrics.pairwise import euclidean_distances
from tqdm import tqdm

from ptfid.utils import local_seed


###########################################
#   Automatic Grid Search
###########################################
def set_grid(data):  # noqa: D103
    import numpy as np

    # find min max
    dim = len(data[0])
    mins = np.array([])
    maxs = np.array([])
    for dims in range(dim):
        mins = np.append(mins, min(data[:, dims]))
        maxs = np.append(maxs, max(data[:, dims]))

    # set grid
    # 2 dimensional data
    if len(mins) == 2:
        xval = np.linspace(mins[0], maxs[0], 1000)
        yval = np.linspace(mins[1], maxs[1], 1000)
        positions = np.array([[u, v] for u in xval for v in yval])
    # 3 dimensional data
    elif len(mins) == 3:
        xval = np.linspace(mins[0], maxs[0], 100)
        yval = np.linspace(mins[1], maxs[1], 100)
        zval = np.linspace(mins[2], maxs[2], 100)
        positions = np.array([[u, v, k] for u in xval for v in yval for k in zval])

    return positions


###########################################
#   KDE with Epanechinikov Kernel
###########################################
def compact_KDE(data, position, h, kernel='cosine'):  # noqa: D103
    # compact kernel options = {"epanechinikov", "cosine"}
    p_hat = np.array([])
    dist = euclidean_distances(position, data)

    # Epanechinikov kernel
    if kernel == 'epanechinikov':
        for iloop in range(len(dist)):
            sample_score = dist[iloop][np.where(dist[iloop] ** 2 <= (h**2))]
            p_hat = np.append(
                p_hat,
                (1 / len(data))
                * ((3 / (4 * h)) ** len(data[0]))
                * ((len(sample_score)) - np.sum(sample_score / (h**2))),
            )
        return p_hat

    # Cosine kernel
    elif kernel == 'cosine':
        for iloop in range(len(dist)):
            sample_score = dist[iloop][np.where(dist[iloop] ** 2 <= (h**2))]
            p_hat = np.append(
                p_hat,
                (1 / len(data))
                * ((np.pi / (4 * h)) ** len(data[0]))
                * np.sum(np.cos((np.pi / 2) * (sample_score / h))),
            )
        return p_hat


###########################################
# Confidence Band
###########################################
def confband_est(data, h, alpha=0.1, kernel='cosine', p_val=True, repeat=10):  # noqa: D103
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    if len(data[0]) <= 3:
        positions = set_grid(data)
    else:
        positions = data

    # p_hat
    p_hat = compact_KDE(data, positions, h, kernel=kernel)

    # p_tilde
    theta_star = np.array([])
    for _ in range(repeat):
        data_bs = data[np.random.choice(np.arange(len(data)), size=len(data), replace=True)]
        p_tilde = compact_KDE(data_bs, positions, h, kernel=kernel)

        # theta
        theta_star = np.append(theta_star, np.sqrt(len(data)) * np.max(np.abs(p_hat - p_tilde)))

    # q_alpha
    if len(theta_star) == 0:
        q_alpha = 0
    else:
        q_alpha = np.quantile(theta_star, 1 - alpha)

    # confidence band
    if p_val:
        return q_alpha / np.sqrt(len(data)), p_hat
    else:
        return q_alpha / np.sqrt(len(data))


###########################################
# BandWidth estimator
###########################################
def bandwidth_est(data, bandwidth_list=tuple(), confidence_band=False, kernel='cosine', alpha=0.1):  # noqa: D103
    # non-compact kernel list:
    #   {'gaussian','exponential'} | compact kernel list = {'tophat','epanechnikov','linear','cosine'}
    if not isinstance(data, np.ndarray):
        data = np.asarray(data)

    # estimate bandwidth candidates with "balloon estimator" (variable-bandwidth estimation)
    if len(bandwidth_list) == 0:
        dist = distance.cdist(data, data, metric='euclidean')
        dist = dist[:-50,]
        for iloop in range(len(dist)):
            if iloop == 0:
                balloon_est = np.array(sorted(dist[iloop, (iloop + 1) :])[50 - 1])
            else:
                balloon_est = np.append(balloon_est, sorted(dist[iloop, (iloop + 1) :])[50 - 1])
        balloon_est = sorted(balloon_est)
        bandwidth_list = balloon_est[int(len(balloon_est) * 0.05) - 1]  # top 5% estimated bandwidth
        bandwidth_list = np.append(
            bandwidth_list, balloon_est[int(len(balloon_est) * 0.2) - 1]
        )  # top 20% estimated bandwidth
        bandwidth_list = np.append(
            bandwidth_list, balloon_est[int(len(balloon_est) * 0.35) - 1]
        )  # top 35% estimated bandwidth
        bandwidth_list = np.append(
            bandwidth_list, balloon_est[int(len(balloon_est) * 0.5) - 1]
        )  # median estimated bandwidth
        bandwidth_list = np.append(
            bandwidth_list, balloon_est[int(len(balloon_est) * 0.65) - 1]
        )  # top 65% estimated bandwidth
        bandwidth_list = np.append(
            bandwidth_list, balloon_est[int(len(balloon_est) * 0.8) - 1]
        )  # top 80% estimated bandwidth
        bandwidth_list = np.append(
            bandwidth_list, balloon_est[int(len(balloon_est) * 0.95) - 1]
        )  # top 95% estimated bandwidth

    # estimate bandwidth
    n_h0 = np.array([])
    s_h0 = np.array([])
    cn_list = np.array([])
    for h in tqdm(bandwidth_list):
        # confidence band & p_hat
        cn = confband_est(data, h, alpha=alpha, kernel=kernel, p_val=False)
        cn_list = np.append(cn_list, cn)

        grid = set_grid(data)
        # filtration
        p_hat = compact_KDE(data, grid, h, kernel=kernel)
        PD = gudhi.CubicalComplex(
            dimensions=[round(len(grid) ** (1 / grid.shape[1])), round(len(grid) ** (1 / grid.shape[1]))],
            top_dimensional_cells=-p_hat,
        ).persistence()

        # measure life length of all homology
        l_h0 = np.array([])
        for iloop in range(len(PD)):
            if PD[iloop][0] == 0 and np.abs(PD[iloop][1][1] - PD[iloop][1][0]) != float('inf'):
                l_h0 = np.append(l_h0, np.abs(PD[iloop][1][1] - PD[iloop][1][0]))

        # N(h)
        n_h0 = np.append(n_h0, sum(l_h0 > cn) + 1)

        # S(h)
        S_h0 = l_h0 - cn
        s_h0 = np.append(s_h0, sum(list(filter(lambda S_h0: S_h0 > 0, S_h0))))
        print('bandwidth: ', h, ', N_0(h): ', n_h0[-1], ', S_0(h): ', s_h0[-1], ', cn: ', cn)

    try:
        if sum(s_h0 == max(s_h0)) == 1:
            if confidence_band:
                return bandwidth_list[s_h0.tolist().index(max(s_h0))], cn_list[s_h0.tolist().index(max(s_h0))]
            else:
                return bandwidth_list[s_h0.tolist().index(max(s_h0))]
        else:
            return bandwidth_list[0]
    except Exception as e:
        print(e)
        raise


###########################################
#   Top P&R
###########################################
@torch.no_grad()
def calculate_toppr(
    features1: np.ndarray,
    features2: np.ndarray,
    alpha: float = 0.1,
    kernel: str = 'cosine',
    random_proj: bool = True,
    f1_score: bool = True,
    feat1_is_real: bool = True,
    seed: int = 0,
) -> dict[str, float]:
    """Compute Top Precision and Recall.

    Args:
    ----
        features1 (n, d): input real features
        features2 (n, d): input fake features
        alpha (float): significance level alpha in confidence band estimation. Default: 0.1.
        kernel (str): kernel for KDE. Default: 'cosine'.
        random_proj (bool): If true, it will add linear layer from Pytorch library for random projection. However, If
            the dimension of the feature is lower than 32, even though random_proj is True, random projection will not
            be turned on. Default: True.
        f1_score (bool): If True, it caculates f1 score for getting a 1-score evaluation. Default: True.
        feat1_is_real (bool, optional): Switch whether `features1` or `features2` is extracted from real samples.
            Default: True.
        seed (int, optional): Random seed. Default: 0.

    Returns:
    -------
        dict[str, float]: fidelity, diversity and (opitionally f1 score.)

    """
    real_features, fake_features = (features1, features2) if feat1_is_real else (features2, features1)

    with local_seed(seed):
        # match data format for random projection
        if not torch.is_tensor(real_features):
            real_features = torch.tensor(real_features, dtype=torch.float32)
        if not torch.is_tensor(fake_features):
            fake_features = torch.tensor(fake_features, dtype=torch.float32)

        # random projection
        if random_proj and real_features.size(1) > 32:
            projection = torch.nn.Linear(real_features.size(1), 32, bias=False).eval()
            torch.nn.init.xavier_normal_(projection.weight)
            for param in projection.parameters():
                param.requires_grad_(False)
            real_features = projection(real_features)
            fake_features = projection(fake_features)

        # to numpy
        real_features = real_features.cpu().numpy()
        fake_features = fake_features.cpu().numpy()

        # use bandwidth estimator to calculate Top P&R
        if len(real_features[0]) <= 3:
            bandwidth_r, c_r = bandwidth_est(real_features, bandwidth_list=[], confidence_band=True, alpha=alpha)
            bandwidth_f, c_g = bandwidth_est(fake_features, bandwidth_list=[], confidence_band=True, alpha=alpha)

        # use balloon estimator to calculate Top P&R
        else:
            k = len(real_features[0]) * 5
            dist = distance.cdist(real_features, real_features, metric='euclidean')
            dist = dist[:-k,]
            for iloop in range(len(dist)):
                if iloop == 0:
                    balloon_est = np.array(sorted(dist[iloop, (iloop + 1) :])[k - 1])
                else:
                    balloon_est = np.append(balloon_est, sorted(dist[iloop, (iloop + 1) :])[k - 1])
            balloon_est = sorted(balloon_est)
            bandwidth_r = balloon_est[int(len(balloon_est) * 0.5) - 1]  # median balloon estimated bandwidth

            # estimate bandwidth for fake with "balloon estimator" (variable-bandwidth estimation)
            dist = distance.cdist(fake_features, fake_features, metric='euclidean')
            dist = dist[:-k,]
            for iloop in range(len(dist)):
                if iloop == 0:
                    balloon_est = np.array(sorted(dist[iloop, (iloop + 1) :])[k - 1])
                else:
                    balloon_est = np.append(balloon_est, sorted(dist[iloop, (iloop + 1) :])[k - 1])
            balloon_est = sorted(balloon_est)
            bandwidth_f = balloon_est[int(len(balloon_est) * 0.5) - 1]  # median balloon estimated bandwidth

            # estimation of confidence band and manifold
            c_r, score_rr = confband_est(data=real_features, h=bandwidth_r, alpha=alpha, kernel=kernel, p_val=True)
            c_g, score_gg = confband_est(data=fake_features, h=bandwidth_f, alpha=alpha, kernel=kernel, p_val=True)

        # count significant real & fake samples
        num_real = np.sum(score_rr > c_r)
        num_fake = np.sum(score_gg > c_g)

        # count significant fake samples on real manifold
        score_rg = compact_KDE(fake_features, real_features, bandwidth_f, kernel=kernel)
        inter_r = np.sum((score_rr > c_r) * (score_rg > c_g))

        # count significant real samples on fake manifold
        score_gr = compact_KDE(real_features, fake_features, bandwidth_r, kernel=kernel)
        inter_g = np.sum((score_gg > c_g) * (score_gr > c_r))

    # topological precision and recall
    t_precision = inter_g / num_fake
    t_recall = inter_r / num_real

    # top f1-score
    if f1_score:
        if t_precision > 0.0001 and t_recall > 0.0001:
            F1_score = 2 / ((1 / t_precision) + (1 / t_recall))
        else:
            F1_score = 0
        return dict(top_precision=float(t_precision), top_recall=float(t_recall), top_f1=float(F1_score))
    else:
        return dict(top_precision=float(t_precision), top_recall=float(t_recall))
