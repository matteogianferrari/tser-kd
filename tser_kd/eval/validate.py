import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from snntorch import functional as SF
from snntorch import utils
from tser_kd.utils import MetricMeter
from tser_kd.dataset import Encoder


@torch.no_grad()
def accuracy(predictions: torch.Tensor, targets: torch.Tensor, top_k: tuple = (1,)) -> list[torch.Tensor]:
    """Computes the top‐k accuracy for the given logits and true labels.

    This function calculates the percentage of correct predictions within the top‐k highest scoring
    classes for each input in the batch. It supports computing multiple k values at once.

    Args:
        predictions: Logits of shape [B, K].
        targets: Ground‐truth class indices of shape [B].
        top_k: A tuple of k values for which to compute accuracy.

    Returns:
        list: A list of accuracy percentages corresponding to each k in `top_k`.
    """
    # Computes the maximum k
    max_k = max(top_k)

    # Retrieves the batch-size B
    B = targets.size(0)

    # Gets the indices of the 'max_k' scores for each sample
    _, pred = predictions.topk(max_k, dim=1, largest=True, sorted=True)

    # Transposes the tensor
    pred = pred.T

    # Compare each of the top‐k indices against the true labels (broadcasting)
    correct = pred.eq(targets.unsqueeze(0))

    res = []
    for k in top_k:
        # Takes the first k rows and count how many are correct
        correct_k = correct[:k].flatten().sum(dtype=torch.float32)

        # Converts to percentage
        res.append(correct_k * (100.0 / B))

    return res


@torch.no_grad()
def accuracy_snn(spikes: torch.Tensor,  targets: torch.Tensor, top_k: tuple = (1,)) -> list[torch.Tensor]:
    """Computes the top‐k accuracy for the given spikes and true labels.

    This function calculates the percentage of correct predictions within the top‐k highest scoring
    classes for each input in the batch. It supports computing multiple k values at once.

    Args:
        spikes: Spike output from network [T, B, K].
        targets: Ground‐truth class indices of shape [B].
        top_k: A tuple of k values for which to compute accuracy.

    Returns:
        list: A list of accuracy percentages corresponding to each k in `top_k`.
    """
    # SF.accuracy_rate(predictions, targets) * 100.0)
    return []


@torch.no_grad()
def run_eval(
    data_loader: DataLoader,
    model: nn.Module,
    criterion: nn.Module,
    device: torch.device,
    encoder: Encoder = None
) -> tuple:
    """Evaluates the model on the given dataset and computes loss and accuracy metrics.

    This function uses PyTorch AMP to perform model's predictions.

    Args:
        data_loader: A PyTorch DataLoader providing (inputs, targets) pairs.
        model: The neural network to evaluate.
        criterion: Loss function used to compute the validation loss.
        device: Device on which to perform computation.
        encoder: Encoder used to convert images into spike trains.

    Returns:
        tuple: A 4-tuple containing:
            - loss.avg: Average loss per example over the entire dataset.
            - top1.avg: Average top‐1 accuracy (%) over the entire dataset.
            - top5.avg: Average top‐5 accuracy (%) over the entire dataset.
            - batch_time.avg: Average elapsed time (in seconds) per batch during evaluation.
    """
    # Device check
    if device == 'cpu':
        device_type = 'cpu'
    else:
        device_type = 'cuda'

    # Puts the model in evaluation mode
    model.eval()

    # Metrics recorders
    batch_time = MetricMeter()
    loss = MetricMeter()
    top1 = MetricMeter()  # Validation accuracy
    top5 = MetricMeter()

    # Evaluates the model over the entire dataset
    for inputs, targets in data_loader:
        # Starts the timer
        ref_time = time.time()

        # Offload the inputs and targets to the desired device with asynchronous operation
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        # CUDA automatic mixed precision
        with torch.amp.autocast(device_type=device_type):
            # Checks if the model is an ANN or a SNN
            if encoder is not None:
                # SNN
                # Encodes the data with the specified encoder type
                inputs = encoder(inputs)
                # [T, B, C, H, W]

                # Resets LIF neurons' hidden states
                utils.reset(model)

                # Computes the model's predictions
                logits = model(inputs)

                # [T, B, K]

                # Computes the loss value between predictions and targets
                loss_val = criterion(logits, targets)
            else:
                # ANN
                # Computes the model's predictions
                logits = model(inputs)

                # Computes the loss value between predictions and targets
                loss_val = criterion(logits, targets)

        # Checks if the model is an ANN or a SNN
        if encoder is not None:
            # SNN
            # Computes the accuracy of the model [B, C, H, W]
            acc1, acc5 = accuracy(predictions=logits, targets=targets, top_k=(1, 5))
        else:
            # ANN
            # Computes the accuracy of the model
            acc1, acc5 = accuracy(predictions=logits, targets=targets, top_k=(1, 5))

        # Retrieves the batch-size
        B = targets.size(0)

        # Updates the metrics
        loss.update(val=loss_val.item(), n=B)
        top1.update(val=acc1.item(), n=B)
        top5.update(val=acc5.item(), n=B)
        batch_time.update(val=time.time() - ref_time)

    return loss.avg, top1.avg, top5.avg, batch_time.avg
