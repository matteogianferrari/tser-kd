"""Teacher Package.

Modules:
    teacher: Defines the 'make_teacher_model' function used to constructs and returns
        a teacher model based on the specified architecture.
"""

from .teacher import make_teacher_model
from .resnet import ResNetBlock, ResNet19


__all__ = [
    "make_teacher_model",
    "ResNetBlock",
    "ResNet19",
]
