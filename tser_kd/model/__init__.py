"""Model Package.

Sub-Packages:
    teacher: Defines functions related to teacher models.
"""

from .layer import conv3x3, conv1x1
from .loss import TSCELoss, TSKLLoss, EntropyReg, TSERKDLoss
from .transfer import transfer_weights_resnet18_resnet19, transfer_weights_resnet18_sresnet18
from .transfer import transfer_weights_resnet19_sresnet19

__all__ = [
    "conv3x3",
    "conv1x1",
    "TSCELoss",
    "TSKLLoss",
    "EntropyReg",
    "TSERKDLoss",
    "transfer_weights_resnet18_resnet19",
    "transfer_weights_resnet18_sresnet18",
    "transfer_weights_resnet19_sresnet19",
]
