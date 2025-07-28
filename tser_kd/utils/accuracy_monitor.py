import torch
import torch.nn as nn


class AccuracyMonitor:
    """Monitor to save the model that achieve the best validation accuracy during training.

    Monitors the validation accuracy at each epoch and saves model checkpoints when an improvement
    is observed.

    Attributes:
        path: File path to save the best model checkpoint.
        epoch_i_max: Index of the epoch with the highest validation accuracy so far.
        best_acc: Best (highest) validation accuracy observed so far.
    """

    def __init__(self, path: str = 'best_accuracy.pth') -> None:
        """Initializes the BestAccuracy.

        Args:
            path: File path where the model with the best accuracy will be saved.
        """
        self.path = path

        self.epoch_i_max = -1
        self.best_acc = None

    def __call__(self, val_acc, epoch_i: int, model: nn.Module) -> None:
        """Checks whether the validation accuracy has improved.

        Args:
            val_acc: The validation accuracy for the current epoch.
            epoch_i: The current epoch index (zero-based).
            model: The model being trained, whose state_dict will be saved/loaded.
        """
        # Edge case: First epoch and nothing to compare with
        if self.best_acc is None:
            # Saves the validation accuracy and epoch
            self.best_acc = val_acc
            self.epoch_i_max = epoch_i

            # Saves a checkpoint of the model's parameters
            torch.save(model.state_dict(), self.path)
        elif val_acc >= self.best_acc:
            # Improvement has been seen, saves the validation accuracy and epoch
            self.best_acc = val_acc
            self.epoch_i_max = epoch_i

            # Saves a checkpoint of the model's parameters
            torch.save(model.state_dict(), self.path)
