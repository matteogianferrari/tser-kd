"""Eval Package.

Modules:
    validate: Defines functions used to evaluate the model on a dataset and computes loss and accuracy metrics.
"""

from .validate import accuracy, accuracy_snn, run_eval

__all__ = [
    "accuracy",
    "accuracy_snn",
    "run_eval",
]
