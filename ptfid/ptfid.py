"""main."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from ptfid import utils
from ptfid.config import FeatureExtractionParameters, MetricFlags, MetricParameters
from ptfid.const import FEATURE_PARAMS_DEFAULT, METRIC_FLAGS_DEFAULT, METRIC_PARAMS_DEFAULT
from ptfid.core import calculate_metrics_from_features
from ptfid.data.dataset import create_dataset
from ptfid.feature import get_feature_extractor
from ptfid.logger import Timer, get_logger
from ptfid.utils import InMemoryFeatureCache, local_deterministic


@torch.no_grad()
def get_features(dataset, model: torch.nn.Module, device: torch.device, progress: bool = True) -> np.ndarray:
    """Extractor features from all images inside the dataset.

    Args:
    ----
        dataset (DataLoader): The dataset.
        model (torch.nn.Module): feature extractor model.
        device (torch.device): device
        progress (boo, optional): show progress bar. Defaults to False

    Returns:
    -------
        np.ndarray: the extracted features.

    """
    features = []
    with local_deterministic():
        for batch in tqdm(dataset, disable=not progress, bar_format='{l_bar}{bar:15}{r_bar}'):
            image = batch['image'].to(device)
            output = model(image)
            features.append(output.cpu().numpy())
    features = np.concatenate(features)
    return features


def calculate_metrics_from_folders(
    dataset_dir1: str,
    dataset_dir2: str,
    # Metric flags.
    metrics: MetricFlags = METRIC_FLAGS_DEFAULT,
    # Metric parameters.
    metric_params: MetricParameters = METRIC_PARAMS_DEFAULT,
    # Feature extraction.
    feature_params: FeatureExtractionParameters = FEATURE_PARAMS_DEFAULT,
    # Other arguments.
    result_file: str | None = None,
) -> dict[str, float]:
    """Compute metrics between two folders.

    Args:
    ----
        dataset_dir1 (str): Dir to dataset.
        dataset_dir2 (str): Dir to dataset.
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
        feature_params (FeatureExtractionParameters, optional): Feature extraction parameters. Defaults to \
            FeatureExtractionParameters(). \
            - `feature_extractor`: Name of the feature extractor or a `nn.Module` object. If the given object is a \
                `nn.Module`, `image_size` argument is also required. Defaults to `'inceptionv3'`. \
            - resizer: Resize method name. Defaults to `'tensorflow'`. \
                - `'tensorflow'`: tensorflow v1 compatible resize. Prefered when using 'inceptionv3' as the feature \
                    extractor. \
                - `'clean'`: Resize presented in Clean-FID. Recommended for most cases. \
                - `'torch'`: Resize image in `torch.Tensor` using `torchvision.transforms.v2.functional.resize()`. \
                - `'pillow'`: Resize image using `Image.Image.resize()`. \
            - normalizer: Normalization method name. Defaults to `'inception'`. \
                - `'inception'`: (x - 0.5) / 0.5. Scale [0, 1] to [-1, 1]. \
                - `'imagenet'`: Normalize data using ImageNet mean std. \
                - `'openai'`: Normalize data using mean std used to train CLIP models. \
                - `'custom'`: Use custom mean std. \
            - `batch_size`: Batch size. Defaults to `32`. \
            - `mean`: Mean for custom normalize. Defaults to `None`. \
            - `std`: Std for custom normalize. Defaults to `None`. \
            - `interpolation`: Interpolation mode. Defaults to `'bicubic'`. \
            - `num_workers`: Number of workers for DataLoader. Defaults to `8`. \
            - `device`: Device to run feature extraction. Defaults to `'cuda'`. \
            - `cache_features`: Cache real features to skip feature extraction on later calls. Useful when computing \
                scores on different generated image sets. Defaults to `True`. \
            - `dataset1_is_real`: Switch real dataset argument. Defaults to `True`. \
            - `image_size`: Input size of feature extractor. Required when feature_extractor argument is a nn.Module. \
                Ignored otherwise. Defaults to `None`.
        result_file (str, optional): JSON file to save results. Defaults to `None`.

    Returns:
    -------
        dict[str, float]: Scores.

    """
    logger = get_logger()
    timer = Timer()

    real_dataset_dir, fake_dataset_dir = (
        (dataset_dir1, dataset_dir2) if feature_params.dataset1_is_real else (dataset_dir2, dataset_dir1)
    )

    device = feature_params.device
    feature_extractor = feature_params.feature_extractor

    # models supported by `ptfid`.
    if isinstance(feature_extractor, str):
        model, image_size = get_feature_extractor(feature_extractor, device=device)
    # nn.Module input as `feature_extraction` argument.
    elif isinstance(feature_extractor, nn.Module):
        image_size = feature_params.image_size

        # `image_size` argument is required in this case.
        if image_size is None:
            raise Exception('If passing a nn.Module as `feature_extractor`, `image_size` must be given.')

        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if len(image_size) != 2:
            raise Exception('`image_size` must be a sequence of 2 elements or an integer object.')

        # ensure the model is ready for eval usage.
        model = feature_extractor.to(device=device)
        model.eval()
        model.requires_grad_(False)

    # Check output size meets requirements.
    _output: torch.Tensor = model(torch.randn(1, 3, *image_size, device=device))
    if _output.ndim != 2:
        raise Exception(f'The output vector must have 2 dimensions. Got {_output.ndim}.')

    logger.info('Extract features from real dataset...')
    timer.start()

    real_features = None
    # load from cache.
    # We only cache real features.
    if feature_params.cache_features:
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
            batch_size=feature_params.batch_size,
            image_size=image_size,
            resize_name=feature_params.resizer,
            normalize_name=feature_params.normalizer,
            mean=feature_params.mean,
            std=feature_params.std,
            interpolation=feature_params.interpolation,
            num_workers=feature_params.num_workers,
            pin_memory=device == 'cuda',
        )

        real_features = get_features(dataset=real_dataset, model=model, device=device)

        if feature_params.cache_features:
            logger.debug('  Caching real features.')
            InMemoryFeatureCache.set(real_dataset_dir, feature_extractor, real_features)

    duration = timer.done()
    logger.info(f'Done. Duration: {duration}, Feature shape: {real_features.shape}')
    logger.info('Extract features from fake dataset...')
    timer.start()

    fake_dataset = create_dataset(
        image_folder=fake_dataset_dir,
        batch_size=feature_params.batch_size,
        image_size=image_size,
        resize_name=feature_params.resizer,
        normalize_name=feature_params.normalizer,
        mean=feature_params.mean,
        std=feature_params.std,
        interpolation=feature_params.interpolation,
        num_workers=feature_params.num_workers,
        pin_memory=device == 'cuda',
    )

    fake_features = get_features(dataset=fake_dataset, model=model, device=device)

    duration = timer.done()
    logger.info(f'Done. Duration: {duration}, Feature shape: {fake_features.shape}')
    logger.info('Compute metrics...')
    timer.start()

    # We always know that feature1 is real due to feature_params.dataset1_is_real
    metric_params.feat1_is_real = True
    results = calculate_metrics_from_features(
        features1=real_features,
        features2=fake_features,
        metrics=metrics,
        metric_params=metric_params,
    )

    duration = timer.done()
    logger.info(f'Done. Duration: {duration}')

    # Keeping this functionality out from `calculate_metrics_from_features()`, which should focus only on metric
    # computation.
    if result_file is not None:
        utils.save_json(obj=results, filename=result_file, avoid_overwrite=True)

    return results
