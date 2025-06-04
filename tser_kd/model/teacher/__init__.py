"""Teacher Package.

Modules:
    teacher: Defines the 'make_teacher_model' function used to constructs and returns
        a teacher model based on the specified architecture.
"""

from .teacher import make_teacher_model

__all__ = [
    "make_teacher_model",
]
