"""Dataset Package.

Modules:
    cutout: Defines the 'CutOut' class used to randomly masks out one or more square patches from an image.
    cifar10: Defines the 'load_cifar10_data' function used to load the CIFAR-10 training and validation datasets.
"""

from .cutout import CutOut
from .cifar10 import load_cifar10_data

__all__ = [
    "CutOut",
    "load_cifar10_data",
]
