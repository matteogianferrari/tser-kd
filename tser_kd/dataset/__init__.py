"""Dataset Package.

Modules:
    cutout: Defines the CutOut class used to randomly masks out one or more square patches from an image.
"""

from .cutout import CutOut
from .cifar10 import load_cifar10_data

__all__ = [
    "CutOut",
    "load_cifar10_data",
]
