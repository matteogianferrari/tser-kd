"""Student Package.

Modules:
    layer: Defines custom layers used to build student ResNet architectures.
"""

from .snn_layer import LayerTWrapper, LIFTWrapper
from .student import SCNN_S, SCNN_T, make_student_model

__all__ = [
    "LayerTWrapper",
    "LIFTWrapper",
    "SCNN_S",
    "SCNN_T",
    "make_student_model",
]
