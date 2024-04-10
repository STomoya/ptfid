"""main."""

from __future__ import annotations

from typing import Literal

import numpy as np
import torch
from tqdm import tqdm

from ptfid import utils
from ptfid.core import compute_metrics_from_features
from ptfid.data.dataset import create_dataset
from ptfid.feature import get_feature_extractor
from ptfid.logger import Timer, get_logger
from ptfid.utils import InMemoryFeatureCache


@torch.no_grad()
def get_features(dataset, model: torch.nn.Module, device: torch.device, progress: bool = True) -> np.ndarray:
    """Extractor features from all images inside the dataset.

    Args:
    ----
        dataset (DataLoader): The dataset.
        model (torch.nn.Module): feature extractor model.
        device (torch.device): device
        progress (boo, optional): show progress bar. Default: False

    Returns:
    -------
        np.ndarray: the extracted features.

    """
    features = []
    for batch in tqdm(dataset, disable=not progress, bar_format='{l_bar}{bar:15}{r_bar}'):
        image = batch['image'].to(device)
        output = model(image)
        features.append(output.cpu().numpy())
    features = np.concatenate(features)
    return features


def calculate_metrics_from_folders(
    dataset_dir1: str,
    dataset_dir2: str,
    feature_extractor: str = 'inceptionv3',
    # Metric flags.
    fid: bool = True,
    kid: bool = False,
    pr: bool = False,
    dc: bool = False,
    pppr: bool = False,
    toppr: bool = False,
    # Metric parameters.
    eps: float = 1e-12,
    fid_compute_method: Literal['original', 'efficient', 'gpu'] = 'efficient',
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
    seed: int = 0,
    # Dataset parameters.
    resizer: Literal['clean', 'torch', 'tensorflow', 'pillow'] = 'tensorflow',
    normalizer: Literal['imagenet', 'openai', 'inception', 'custom'] = 'inception',
    batch_size: int = 32,
    mean: tuple[float, ...] | None = None,
    std: tuple[float, ...] | None = None,
    num_workers: int = 8,
    # Feature extraction.
    device: Literal['cpu', 'cuda'] = 'cuda',
    cache_features: bool = True,
    dataset1_is_real: bool = True,
    # Other arguments.
    result_file: str | None = None,
) -> dict[str, float]:
    """Compute metrics between two folders.

    Args:
    ----
        dataset_dir1 (str): Dir to dataset.
        dataset_dir2 (str): Dir to dataset.
        feature_extractor (str, optional): Name of the feature extractor. Default: 'inceptionv3'.
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
        seed (int, optional): Random seed. Default: 0.
        resizer (Literal['clean', 'torch', 'tensorflow', 'pillow'], optional): Resize method name.
            - 'tensorflow': tensorflow v1 compatible resize. Prefered when using 'inceptionv3' as the feature extractor.
            - 'clean': Resize presented in Clean-FID. Recommended for most cases.
            - 'torch': Resize image in `torch.Tensor` using `torchvision.transforms.v2.functional.resize()`.
            - 'pillow': Resize image using `Image.Image.resize()`.
            Default: 'tensorflow'.

        normalizer (Literal['imagenet', 'openai', 'inception', 'custom'], optional): Normalization method name.
            - 'inception': (x - 0.5) / 0.5. Scale [0, 1] to [-1, 1].
            - 'imagenet': Normalize data using ImageNet mean std.
            - 'openai': Normalize data using mean std used to train CLIP models.
            - 'custom': Use custom mean std.
            Default: 'inception'.

        batch_size (int, optional): Batch size. Default: 32.
        mean (list[float] | float | None, optional): Mean for custom normalize. Default: None.
        std (list[float] | float | None, optional): Std for custom normalize. Default: None.
        num_workers (int, optional): Number of workers for DataLoader. Default: 8.
        device (Literal['cpu', 'cuda'], optional): Device to run feature extraction. Default: 'cuda'.
        cache_features (bool, optional): Cache real features to skip feature extraction on later calls. Useful when
            computing scores on different generated image sets. Default: True.
        dataset1_is_real (bool, optional): Switch real dataset argument. Default: True.
        result_file (str, optional): JSON file to save results.

    Returns:
    -------
        dict[str, float]: Scores.

    """
    logger = get_logger()
    timer = Timer()

    real_dataset_dir, fake_dataset_dir = (
        (dataset_dir1, dataset_dir2) if dataset1_is_real else (dataset_dir2, dataset_dir1)
    )

    model, image_size = get_feature_extractor(feature_extractor, device=device)

    logger.info('Extract features from real dataset...')
    timer.start()

    real_features = None
    # load from cache.
    # We only cache real features.
    if cache_features:
        logger.debug('  Looking for cached features for real dataset...')

        real_features = InMemoryFeatureCache.get(real_dataset_dir, feature_extractor)

        if real_features is None:
            logger.debug(
                '    Failed. Falling back to extracting features. This msg is expected if this is the first call.'
            )
        else:
            logger.debug('    Success. Skipping feature extraction on real dataset.')

    if real_features is None:
        # extract features from real dataset
        real_dataset = create_dataset(
            image_folder=real_dataset_dir,
            batch_size=batch_size,
            image_size=image_size,
            resize_name=resizer,
            normalize_name=normalizer,
            mean=mean,
            std=std,
            num_workers=num_workers,
            pin_memory=device == 'cuda',
        )

        real_features = get_features(dataset=real_dataset, model=model, device=device)

        if cache_features:
            logger.debug('  Caching real features.')
            InMemoryFeatureCache.set(real_dataset_dir, feature_extractor, real_features)

    duration = timer.done()
    logger.info(f'Done. Duration: {duration}, Feature shape: {real_features.shape}')
    logger.info('Extract features from fake dataset...')
    timer.start()

    fake_dataset = create_dataset(
        image_folder=fake_dataset_dir,
        batch_size=batch_size,
        image_size=image_size,
        resize_name=resizer,
        normalize_name=normalizer,
        mean=mean,
        std=std,
        num_workers=num_workers,
        pin_memory=device == 'cuda',
    )

    fake_features = get_features(dataset=fake_dataset, model=model, device=device)

    duration = timer.done()
    logger.info(f'Done. Duration: {duration}, Feature shape: {fake_features.shape}')
    logger.info('Compute metrics...')
    timer.start()

    results = compute_metrics_from_features(
        features1=real_features,
        features2=fake_features,
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
        feat1_is_real=True,
        seed=seed,
    )

    duration = timer.done()
    logger.info(f'Done. Duration: {duration}')

    # Keeping this functionality out from `compute_metrics_from_features()`, which should focus only on metric
    # computation.
    if result_file is not None:
        utils.save_json(obj=results, filename=result_file, avoid_overwrite=True)

    return results
