import torch
import torch.nn as nn


def conv3x3(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1) -> nn.Conv2d:
    """Pre-configured 2D convolution to use in ResNet architectures.

    Args:
        in_channels: Number of input channels C_in.
        out_channels: Number of kernels C_out.
        kernel_size: Size of the kernels.
        stride: Stride to apply over the spatial dimensionality.

    Returns:
        nn.Conv2d: A pre-configured 2D convolution.
    """
    return nn.Conv2d(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=1,
        bias=False
    )
