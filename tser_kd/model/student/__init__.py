"""Student Package.

Modules:
    layer: Defines custom layers used to build student ResNet architectures.
"""

from .layer import conv3x3, conv1x1, TWrapLayer, TDBatchNorm2d

__all__ = [
    "conv3x3",
    "conv1x1",
    "TDBatchNorm2d",
    "TWrapLayer",
]
