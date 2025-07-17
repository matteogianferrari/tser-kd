import math
from torch.optim.lr_scheduler import _LRScheduler
import torch.optim as optim


class WarmupCosineLR(_LRScheduler):
    """LR scheduler that combines a linear warm-up phase with a cosine decay phase.

    The scheduler behaves differently based on the phase:
    - Warm-up phase: The LR is linearly interpolated from a 'base_lr' up to a 'max_lr'.
        This helps the optimizers stabilize during the first few epochs.
    - Cosine decay phase: The LR follows a cosine annealing curve from 'max_lr' for 'total_epochs'.

    Attributes:
        warmup_epochs: Number of epochs for the linear warm‑up period.
        total_epochs: Total number of training epochs.
        base_lr: Starting LR at epoch 0.
        max_lr: Peak LR reached at the end of the warm‑up phase.
        min_lr: Minimal LR reached at the end of cosine decay and held until training finishes.
    """

    def __init__(
        self,
        optimizer: optim,
        warmup_epochs: int,
        total_epochs: int,
        base_lr: float,
        max_lr: float,
        min_lr: float = 0.0,
        last_epoch: int = -1,
    ):
        """Initializes the WarmupCosineLR.

        Args:
            optimizer: Optimizer used during training.
            warmup_epochs: Number of epochs for the linear warm‑up period.
            total_epochs: Total number of training epochs.
            base_lr: Starting LR at epoch 0.
            max_lr: Peak LR reached at the end of the warm‑up phase.
            min_lr: Minimal LR reached at the end of cosine decay and held until training finishes.
            last_epoch: Index of the last processed epoch.  Use the default value when
                constructing the scheduler at the start of training; provide a positive
                number to resume training from a checkpoint.
        """
        super().__init__(optimizer, last_epoch)

        assert warmup_epochs >= 1, "warmup_epochs must be ≥ 1"
        assert total_epochs >= warmup_epochs, "total_epochs must be ≥ warmup_epochs"

        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.min_lr = min_lr

        # Start every param group at base_lr
        for pg in optimizer.param_groups:
            pg["lr"] = base_lr

    def get_lr(self) -> list:
        """Compute the LR for all parameter groups at the upcoming epoch.

        Returns:
            list: A list containing one LR value per optimiser parameter group.
        """
        epoch = self.last_epoch + 1

        # Check scheduling zone
        if epoch < self.warmup_epochs:
            # Linear warmup
            # Computes the fraction to add to the LR
            warm_frac = epoch / self.warmup_epochs

            # Computes the new LR
            lr = self.base_lr + warm_frac * (self.max_lr - self.base_lr)
        else:
            # Cosine decay
            # Computes the fraction to add to the LR
            cos_frac = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            cosine = 0.5 * (1.0 + math.cos(math.pi * cos_frac))

            # Computes the new LR
            lr = self.min_lr + (self.max_lr - self.min_lr) * cosine

        return [lr for _ in self.optimizer.param_groups]
