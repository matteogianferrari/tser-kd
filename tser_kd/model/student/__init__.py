"""Student Package.

Modules:
    layer: Defines custom layers used to build student ResNet architectures.
"""

from .layer import conv3x3, conv1x1

__all__ = [
    "conv3x3",
    "conv1x1",
]
