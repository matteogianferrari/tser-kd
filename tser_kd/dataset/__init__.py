"""Dataset Package.

Modules:
    cutout: Defines the CutOut class used to randomly masks out one or more square patches from an image.
"""

from .cutout import CutOut

__all__ = [
    "CutOut",
]
