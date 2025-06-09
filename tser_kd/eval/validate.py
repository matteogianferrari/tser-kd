import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from snntorch import functional as SF
from snntorch import utils
from tser_kd.utils import MetricMeter
from tser_kd.dataset import Encoder


@torch.no_grad()
def accuracy(logits: torch.Tensor, targets: torch.Tensor, top_k: tuple = (1,)) -> list[torch.Tensor]:
    """Computes the top‐k accuracy for the given logits and true labels.

    This function calculates the percentage of correct predictions within the top‐k highest scoring
    classes for each input in the batch. It supports computing multiple k values at once.

    Args:
        logits: Logits tensor of shape [B, K].
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
    _, pred = logits.topk(max_k, dim=1, largest=True, sorted=True)

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
def accuracy_snn(logits: torch.Tensor,  targets: torch.Tensor, top_k: tuple = (1,)) -> list[torch.Tensor]:
    """Computes the top‐k accuracy for the given SNN logits and true labels.

    This function calculates the percentage of correct predictions within the top‐k highest scoring
    classes for each input in the batch. It supports computing multiple k values at once.
    Each top-k accuracy is computed by averaging over the batch B, then by averaging between time steps T.

    Args:
        logits: Logits output from network of shape [T, B, K].
        targets: Ground‐truth class indices of shape [B].
        top_k: A tuple of k values for which to compute accuracy.

    Returns:
        list: A list of accuracy percentages corresponding to each k in `top_k`.
    """
    # Computes the maximum k
    max_k = max(top_k)

    # Gets the indices of the 'max_k' scores for each sample for each time step
    _, pred = logits.topk(max_k, dim=2, largest=True, sorted=True)

    # Retrieves the dimensions, pred.shape: [T, B, max_k]
    T, B, _ = pred.shape

    # Expands the target to match the predictions shape, targets.shape: [T, B]
    targets = targets.unsqueeze(0).expand(T, B)

    res = []
    for k in top_k:
        # Takes only the first k predictions, pred_k.shape: [T, B, k]
        pred_k = pred[:, :, :k]

        # Compares the predictions against the ground truth, correct_any.shape: [T, B]
        correct_k = pred_k.eq(targets.unsqueeze(-1))
        correct_any = correct_k.any(dim=2).float()

        # Averages the accuracies over the batches, acc_per_t.shape: [T]
        acc_per_t = correct_any.mean(dim=1)

        # Averages over all time steps
        res.append(acc_per_t.mean() * 100.0)

    return res


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
        # inputs.shape: [B, C, H, W]
        inputs = inputs.to(device, non_blocking=True)

        # targets.shape: [B]
        targets = targets.to(device, non_blocking=True)

        # CUDA automatic mixed precision
        with torch.amp.autocast(device_type=device_type):
            # Checks if the model is an ANN or a SNN
            if encoder is not None:
                # SNN
                # Encodes the data with the specified encoder type, inputs.shape: [T, B, C, H, W]
                inputs = encoder(inputs)

                # Resets LIF neurons' hidden states
                utils.reset(model)

                # Computes the model's logits, logits.shape: [T, B, K]
                logits = model(inputs)

                # Computes the loss value between logits and targets
                loss_val = criterion(logits, targets)
            else:
                # ANN
                # Computes the model's logits, logits.shape: [B, K]
                logits = model(inputs)

                # Computes the loss value between logits and targets
                loss_val = criterion(logits, targets)

        # Checks if the model is an ANN or a SNN
        if encoder is not None:
            # SNN
            # Computes the accuracy of the model
            acc1, acc5 = accuracy_snn(logits=logits, targets=targets, top_k=(1, 5))
        else:
            # ANN
            # Computes the accuracy of the model
            acc1, acc5 = accuracy(logits=logits, targets=targets, top_k=(1, 5))

        # Retrieves the batch-size
        B = targets.size(0)

        # Updates the metrics
        loss.update(val=loss_val.item(), n=B)
        top1.update(val=acc1.item(), n=B)
        top5.update(val=acc5.item(), n=B)
        batch_time.update(val=time.time() - ref_time)

    return loss.avg, top1.avg, top5.avg, batch_time.avg
