"""tensorflow compat resize.

from: https://github.com/toshas/torch-fidelity/blob/master/torch_fidelity/interpolate_compat_tensorflow.py
hacked by: Tomoya Sawada
    The output is equivalent with the original implementation. This function does not support 4D tensors because we
    always perform resizing inside the dataset and batched data will never be input. Also, functions in a function is
    moved outside to avoid redundant function object creation.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def _output_size(
    size: tuple[int, int] | None,
    is_tracing: bool,
) -> tuple[int, int]:
    if size is not None:
        if is_tracing:
            return tuple(torch.tensor(i) for i in size)
        else:
            return size


def tf_calculate_resize_scale(  # noqa: D103
    in_size: int, out_size: int | torch.Tensor, align_corners: bool, is_tracing: bool
) -> float:
    if align_corners:
        if is_tracing:
            return (in_size - 1) / (out_size.float() - 1).clamp(min=1)
        else:
            return (in_size - 1) / max(1, out_size - 1)
    else:
        if is_tracing:
            return in_size / out_size.float()
        else:
            return in_size / out_size


def resample_using_grid_sample(  # noqa: D103
    input: torch.Tensor, out_size: tuple[int, int], scale_x: float, scale_y: float
) -> torch.Tensor:
    grid_x = torch.arange(0, out_size[1], 1, dtype=input.dtype, device=input.device)
    grid_x = grid_x * (2 * scale_x / (input.shape[3] - 1)) - 1

    grid_y = torch.arange(0, out_size[0], 1, dtype=input.dtype, device=input.device)
    grid_y = grid_y * (2 * scale_y / (input.shape[2] - 1)) - 1

    grid_x = grid_x.view(1, out_size[1]).repeat(out_size[0], 1)
    grid_y = grid_y.view(out_size[0], 1).repeat(1, out_size[1])

    grid_xy = torch.stack((grid_x, grid_y), dim=-1).unsqueeze(0)

    out = F.grid_sample(
        input,
        grid_xy,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    return out


def resample_manually(input: torch.Tensor, out_size: tuple[int, int], scale_x: float, scale_y: float) -> torch.Tensor:  # noqa: D103
    grid_x = torch.arange(0, out_size[1], 1, dtype=input.dtype, device=input.device)
    grid_x = grid_x * torch.tensor(scale_x, dtype=torch.float32)
    grid_x_lo = grid_x.long()
    grid_x_hi = (grid_x_lo + 1).clamp_max(input.shape[2] - 1)
    grid_dx = grid_x - grid_x_lo.float()

    grid_y = torch.arange(0, out_size[0], 1, dtype=input.dtype, device=input.device)
    grid_y = grid_y * torch.tensor(scale_y, dtype=torch.float32)
    grid_y_lo = grid_y.long()
    grid_y_hi = (grid_y_lo + 1).clamp_max(input.shape[1] - 1)
    grid_dy = grid_y - grid_y_lo.float()

    # could be improved with index_select
    in_00 = input[:, grid_y_lo, :][:, :, grid_x_lo]
    in_01 = input[:, grid_y_lo, :][:, :, grid_x_hi]
    in_10 = input[:, grid_y_hi, :][:, :, grid_x_lo]
    in_11 = input[:, grid_y_hi, :][:, :, grid_x_hi]

    in_0 = in_00 + (in_01 - in_00) * grid_dx.view(1, 1, out_size[1])
    in_1 = in_10 + (in_11 - in_10) * grid_dx.view(1, 1, out_size[1])
    out = in_0 + (in_1 - in_0) * grid_dy.view(1, out_size[0], 1)

    return out


def interpolate_bilinear_2d_like_tensorflow1x(
    input: torch.Tensor, size: tuple[int, int], align_corners: bool = False, method: str = 'slow'
) -> torch.Tensor:
    r"""Down/up samples the input to either the given :attr:`size` or the given :attr:`scale_factor`.

    Epsilon-exact bilinear interpolation as it is implemented in TensorFlow 1.x:
    https://github.com/tensorflow/tensorflow/blob/f66daa493e7383052b2b44def2933f61faf196e0/tensorflow/core/kernels/image_resizer_state.h#L41
    https://github.com/tensorflow/tensorflow/blob/6795a8c3a3678fb805b6a8ba806af77ddfe61628/tensorflow/core/kernels/resize_bilinear_op.cc#L85
    as per proposal:
    https://github.com/pytorch/pytorch/issues/10604#issuecomment-465783319

    Related materials:
    https://hackernoon.com/how-tensorflows-tf-image-resize-stole-60-days-of-my-life-aba5eb093f35
    https://jricheimer.github.io/tensorflow/2019/02/11/resize-confusion/
    https://machinethink.net/blog/coreml-upsampling/

    Currently only 2D spatial sampling is supported, i.e. expected inputs are 4-D in shape.

    The input dimensions are interpreted in the form:
    `mini-batch x channels x height x width`.

    Args:
    ----
        input (Tensor): the input tensor
        size (Tuple[int, int]): output spatial size.
        align_corners (bool, optional): Same meaning as in TensorFlow 1.x.
        method (str, optional):
            'slow' (1e-4 L_inf error on GPU, bit-exact on CPU, with checkerboard 32x32->299x299), or
            'fast' (1e-3 L_inf error on GPU and CPU, with checkerboard 32x32->299x299)

    """
    if method not in ('slow', 'fast'):
        raise ValueError('how_exact can only be one of "slow", "fast"')

    if input.dim() != 3:
        raise ValueError('input must be a 3-D tensor')

    if not torch.is_floating_point(input):
        raise ValueError('input must be of floating point dtype')

    if not isinstance(size, (int, tuple)) or len(size) != 2:
        raise ValueError('size must be a list or a tuple of two elements')

    is_tracing = torch._C._get_tracing_state()

    _, H, W = input.size()

    out_size = _output_size(size, is_tracing=is_tracing)
    scale_y = tf_calculate_resize_scale(H, out_size[0], align_corners=align_corners, is_tracing=is_tracing)
    scale_x = tf_calculate_resize_scale(W, out_size[1], align_corners=align_corners, is_tracing=is_tracing)

    if method == 'slow':
        out = resample_manually(input, out_size=out_size, scale_x=scale_x, scale_y=scale_y)
    else:
        input = input.unsqueeze(0)  # torch.nn.functional.grid_sample requires batched input...
        out = resample_using_grid_sample(input, out_size=out_size, scale_x=scale_x, scale_y=scale_y)
        out = out.squeeze()

    return out
