"""Training Package.

Modules:
    early_stopping: Defines the EarlyStopping class used to terminate training when validation loss stops improving.
"""

from .early_stopping import EarlyStopping

__all__ = [
    "EarlyStopping",
]
