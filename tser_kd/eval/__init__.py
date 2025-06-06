"""Eval Package.

Modules:
    metric_meter: Defines the 'MetricMeter' class used to tracks a streaming metric using observed values.
    validate: Defines functions used to evaluate the model on a dataset and computes loss and accuracy metrics.
"""

from .metric_meter import MetricMeter
from .validate import *

__all__ = [
    "MetricMeter",
    "accuracy",
    "accuracy_spikes",
    "run_eval",
]
