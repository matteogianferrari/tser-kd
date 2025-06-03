import torch
import torch.nn as nn


class EarlyStopping:
    """Early stopping to terminate training when validation loss stops improving.

    Monitors the validation loss at each epoch and saves model checkpoints when an improvement
    is observed. If no improvement greater than `delta` is seen for `patience` consecutive epochs,
    training stops and the model is reverted to the best checkpoint.

    Attributes:
        patience: Number of epochs to wait after last improvement before stopping.
        delta: Minimum change in validation loss to qualify as an improvement.
        path: File path to save the best model checkpoint.
        early_stop: Flag indicating whether early stopping has been triggered.
        counter: Number of consecutive epochs without significant improvement.
        epoch_i_min: Index of the epoch with the lowest validation loss so far.
        best_score: Best (lowest) validation loss observed so far.
    """

    def __init__(self, patience: int = 1, delta: float = 0.0, path: str = 'model_checkpoint.pth') -> None:
        """Initializes the EarlyStopping.

        Args:
            patience: Number of epochs with no improvement after which training will be stopped.
            delta: Minimum decrease in validation loss to qualify as an improvement.
            path: File path where the best model checkpoint will be saved.
        """
        self.patience = patience
        self.delta = delta
        self.path = path

        self.early_stop = False
        self.counter = 0

        self.epoch_i_min = -1
        self.best_score = None

    def __call__(self, val_loss, epoch_i: int, model: nn.Module) -> bool:
        """Checks whether validation loss has improved.

        Args:
            val_loss: The validation loss for the current epoch.
            epoch_i: The current epoch index (zero-based).
            model: The model being trained, whose state_dict will be saved/loaded.

        Returns:
            bool: True if early stopping has been triggered and training should stop; False otherwise.
        """
        # Edge case: First epoch and nothing to compare with
        if self.best_score is None:
            # Saves the validation loss and epoch
            self.best_score = val_loss
            self.epoch_i_min = epoch_i

            # Saves a checkpoint of the model's parameters
            torch.save(model.state_dict(), self.path)
        elif val_loss > self.best_score - self.delta:
            # No significant improvement seen
            self.counter += 1

            # Checks if the patience has run out
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping triggered! Loading weights from best epoch: {self.epoch_i_min+1}")
                model.load_state_dict(torch.load(self.path))
        else:
            # Improvement has been seen, saves the validation loss and epoch
            self.best_score = val_loss
            self.epoch_i_min = epoch_i

            # Saves a checkpoint of the model's parameters
            torch.save(model.state_dict(), self.path)

            # Resets the counter
            self.counter = 0

        return self.early_stop
