"""Student Package.

Modules:
    layer: Defines custom layers used to build student ResNet architectures.
"""

from .snn_layer import LayerTWrapper, LIFTWrapper
from .student import SCNN, make_student_model

__all__ = [
    "LayerTWrapper",
    "LIFTWrapper",
    "SCNN",
    "make_student_model",
]
